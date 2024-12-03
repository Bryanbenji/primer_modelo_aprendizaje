import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from mlforecast import MLForecast
from window_ops.rolling import rolling_mean, rolling_max, rolling_min
from window_ops.ewm import ewm_mean
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets
from datetime import datetime, timedelta

# Conexión a la base de datos
engine = create_engine('mysql+pymysql://root:Udla1@localhost:3306/wst')

# Consulta SQL proporcionada
query_ticket_materiales = '''
SELECT 
    tm.material_id AS MaterialID,
    tm.fecha,
    tm.cantidad AS Demand,
    tm.stock_anterior,
    tm.stock_actual,
    CASE 
        WHEN tm.serial IS NOT NULL THEN 'Serializado'
        ELSE 'No Serializado'
    END AS TipoMaterial,
    cm.name AS MaterialName,
    cm.description AS MaterialDescription,
    cm.stock_minimo_default AS StockMinimo,
    CASE 
        WHEN tm.serial IS NOT NULL THEN COALESCE(ms_salida.count, 0)
        ELSE COALESCE(ms_sinserial.quantity, 0)
    END AS StockActual
FROM ticket_materiales tm
LEFT JOIN catalogo_materiales cm 
    ON cm.codigo_equipo = tm.codigo_material
LEFT JOIN (
    SELECT
        id AS idserial,
        idmateriales,
        COUNT(*) AS count
    FROM
        materiales_serial
    WHERE
        estado IN ('en uso', 'asignado')
    GROUP BY
        idserial
) ms_salida 
    ON tm.material_id = ms_salida.idserial
LEFT JOIN (
    SELECT
        id AS idsinserial,
        idmateriales,
        SUM(quantity) AS quantity
    FROM
        materiales_sinserial
    GROUP BY
        idsinserial
) ms_sinserial 
    ON tm.material_id = ms_sinserial.idsinserial;
'''

# Cargar los datos
df_ticket_materiales = pd.read_sql(query_ticket_materiales, engine)

# Procesar los datos
df_ticket_materiales['fecha'] = pd.to_datetime(df_ticket_materiales['fecha'])
df_ticket_materiales['stock_anterior'] = df_ticket_materiales['stock_anterior'].astype(float)
df_ticket_materiales['stock_actual'] = df_ticket_materiales['stock_actual'].astype(float)
df_ticket_materiales['Demand'] = df_ticket_materiales['Demand'].astype(float)
df_ticket_materiales['RealUsage'] = df_ticket_materiales['stock_anterior'] - df_ticket_materiales['stock_actual']
df_ticket_materiales['YearMonth'] = df_ticket_materiales['fecha'].dt.to_period('M').dt.to_timestamp()

# Agrupar por mes y material
df_monthly_demand = df_ticket_materiales.groupby(['MaterialID', 'YearMonth'])['Demand'].sum().reset_index()
df_real_usage = df_ticket_materiales.groupby(['MaterialID', 'YearMonth'])['RealUsage'].sum().reset_index()

# Unir con los datos de stock
df = pd.merge(df_monthly_demand, df_ticket_materiales[['MaterialID', 'StockActual', 'MaterialName']].drop_duplicates(), on='MaterialID')
df = pd.merge(df, df_real_usage, on=['MaterialID', 'YearMonth'], how='left')

# Preparar datos para el modelo
df_model = df[['MaterialID', 'YearMonth', 'Demand', 'RealUsage', 'StockActual']].dropna()

# Obtener el mes y año actual
current_date = datetime.now()
validation_start_date = current_date.replace(day=1)  # Primer día del mes actual
train_end_date = validation_start_date - timedelta(days=1)  # Último día del mes anterior

# Convertir a cadenas en formato YYYY-MM-DD
validation_start_date_str = validation_start_date.strftime('%Y-%m-%d')
train_end_date_str = train_end_date.strftime('%Y-%m-%d')

# Dividir los datos en entrenamiento y validación
train_data = df_model[df_model['YearMonth'] <= train_end_date_str]
validation_data = df_model[df_model['YearMonth'] >= validation_start_date_str]

# Validar que no estén vacíos
if train_data.empty:
    raise ValueError("Los datos de entrenamiento o validación están vacíos. Verifica el rango de fechas o la cantidad de datos disponibles.")

# Función para predicción y visualización
def plot(material, lag1, lag2, rolm):
    material_row = df[df['MaterialName'] == material]
    if material_row.empty:
        print(f"Material '{material}' no encontrado en la base de datos.")
        return

    material_id = material_row.iloc[0]['MaterialID']
    train = train_data[train_data['MaterialID'] == material_id]

    if train.empty:
        print(f"No hay suficientes datos históricos para el material '{material}'.")
        return

    if len(train) <= max(lag1, lag2):
        print(f"El material '{material}' no tiene suficientes datos para calcular con los lags seleccionados.")
        return

    # Manejo de valores nulos en la columna objetivo
    train = train.fillna(0)

    models = [
        make_pipeline(SimpleImputer(strategy='mean'), RandomForestRegressor(random_state=0, n_estimators=100)),
        XGBRegressor(random_state=0, n_estimators=100)
    ]

    model = MLForecast(
        models=models,
        freq='M',
        lags=[lag1, lag2],
        lag_transforms={
            lag1: [(rolling_mean, rolm), (rolling_min, rolm), (rolling_max, rolm)],
            lag2: [(ewm_mean, 0.5)],
        },
        num_threads=6
    )

    try:
        model.fit(train, id_col='MaterialID', time_col='YearMonth', target_col='Demand')
    except ValueError as e:
        print(f"Error al entrenar el modelo: {e}")
        return

    h = 3
    predictions = model.predict(h=h)
    predictions['YearMonth'] = pd.date_range(start='2023-12-01', periods=h, freq='M')
    predictions.set_index('YearMonth', inplace=True)

    # Gráfico 1: Demanda histórica y predicción
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    train.set_index('YearMonth')['Demand'].plot(ax=ax1, label='Historical Demand')
    predictions['RandomForestRegressor'].plot(ax=ax1, label='Predicted Demand', linestyle='--')
    ax1.set(title=f'Demand Prediction for {material}', xlabel='Year-Month', ylabel='Demand')
    ax1.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Gráfico 2: Stock histórico y proyección
    stock_actual = material_row.iloc[0]['StockActual']
    projected_stock = stock_actual - predictions['RandomForestRegressor'].cumsum()

    plt.figure(figsize=(12, 6))
    ax2 = plt.gca()
    projected_stock.plot(ax=ax2, label='Projected Stock', linestyle='--', color='orange')
    ax2.axhline(y=50, color='red', linestyle=':', label='Replenishment Threshold')
    ax2.set(title=f'Stock Projection for {material}', xlabel='Year-Month', ylabel='Stock')
    ax2.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Widgets interactivos
material_widget = widgets.Dropdown(options=df['MaterialName'].unique(), description='Material')
lag1_widget = widgets.IntSlider(value=2, min=2, max=10, step=2, description='Lag1')
lag2_widget = widgets.IntSlider(value=8, min=4, max=12, step=2, description='Lag2')
rolm_widget = widgets.IntSlider(value=4, min=2, max=6, step=2, description='Rolm')

ui = widgets.HBox([material_widget, lag1_widget, lag2_widget, rolm_widget])
out = widgets.interactive_output(plot, {'material': material_widget, 'lag1': lag1_widget, 'lag2': lag2_widget, 'rolm': rolm_widget})

display(ui, out)
