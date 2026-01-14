# Detección de fraude con tarjetas de crédito

# Tabla de Contenidos  
## Detección de Fraude con Tarjetas de Crédito

- [Detección de Fraude con Tarjetas de Crédito](#detección-de-fraude-con-tarjetas-de-crédito)
  - [Descripción del problema](#1-descripción-del-problema)
  - [Problemas técnicos del dataset](#2-problemas-técnicos-del-dataset)
    - [Desbalance extremo de clases](#21-desbalance-extremo-de-clases)
    - [Datos ruidosos y comportamiento cambiante](#22-datos-ruidosos-y-comportamiento-cambiante)
    - [Falta de interpretabilidad](#23-falta-de-interpretabilidad)
    - [Datos sensibles y transformados](#24-datos-sensibles-y-transformados)
    - [Costos asimétricos de error](#25-costos-asimétricos-de-error)
  - [¿Por qué es un problema complejo de Machine Learning?](#3-por-qué-es-un-problema-complejo-de-machine-learning)
    - [Métricas de evaluación no triviales](#31-métricas-de-evaluación-no-triviales)
    - [Necesidad de técnicas avanzadas](#32-necesidad-de-técnicas-avanzadas)
    - [Requerimientos de tiempo real](#33-requerimientos-de-tiempo-real)
    - [Evolución continua del fraude](#34-evolución-continua-del-fraude)
    - [Balance entre precisión y experiencia del cliente](#35-balance-entre-precisión-y-experiencia-del-cliente)
  - [Conclusión](#4-conclusión)

## Tabla de Contenidos

- [Entorno Virtual en Python (Ciencia de Datos)](#entorno-virtual-en-python-ciencia-de-datos)
  - [Pasos básicos para usar un entorno virtual](#pasos-básicos-para-usar-un-entorno-virtual)

- [Random Forest para Clasificación](#random-forest-para-clasificación)
  - [Definición](#definición)
  - [¿Cómo funciona Random Forest en clasificación?](#cómo-funciona-random-forest-en-clasificación)
  - [Ventajas principales](#ventajas-principales)
  - [Conclusión](#conclusión)

- [XGBoost: funcionamiento y por qué fue el mejor modelo](#xgboost-funcionamiento-y-por-qué-fue-el-mejor-modelo)

- [Tabla comparativa Random Forest vs XGBoost](#tabla-comparativa-random-forest-vs-xgboost)




## Descripción del problema, retos técnicos y complejidad en Machine Learning

---

## 1. Descripción del problema

La **detección de fraude con tarjetas de crédito** consiste en identificar transacciones financieras que no han sido realizadas por el titular legítimo de la tarjeta, sino por un tercero de manera fraudulenta. El objetivo principal es **clasificar cada transacción como fraudulenta o legítima** en el menor tiempo posible, idealmente en tiempo real, para minimizar pérdidas económicas, proteger al cliente y cumplir con regulaciones financieras.

Este problema se formula típamente como un **problema de clasificación binaria**, donde:
- **Clase 0**: Transacción legítima  
- **Clase 1**: Transacción fraudulenta  

Los datasets suelen contener variables como monto de la transacción, tiempo, localización, tipo de comercio y variables transformadas (por ejemplo, mediante PCA por razones de privacidad).

---

## 2. Problemas técnicos del dataset

### 2.1 Desbalance extremo de clases
Uno de los principales problemas es que el fraude representa un **porcentaje muy bajo del total de transacciones** (usualmente < 1%). Esto genera:
- Modelos sesgados hacia la clase mayoritaria.
- Alta precisión aparente, pero bajo recall en la clase fraudulenta.
- Dificultad para entrenar modelos que realmente detecten fraude.

---

### 2.2 Datos ruidosos y comportamiento cambiante
El comportamiento de los usuarios y de los defraudadores **cambia constantemente**:
- Nuevos patrones de fraude aparecen con el tiempo.
- Transacciones legítimas pueden parecer anómalas.
- El modelo entrenado puede volverse obsoleto (concept drift).

---

### 2.3 Falta de interpretabilidad
Muchos modelos efectivos (Random Forest, XGBoost, Deep Learning) son **difíciles de interpretar**, lo cual es un problema porque:
- El sector financiero exige explicaciones claras de las decisiones.
- Se requieren auditorías y cumplimiento regulatorio.
- Los analistas necesitan justificar por qué una transacción fue bloqueada.

---

### 2.4 Datos sensibles y transformados
Por razones de privacidad:
- Muchas variables están **anonimizadas o transformadas** (ej. PCA).
- Se pierde contexto de negocio.
- Se dificulta el feature engineering y el análisis exploratorio.

---

### 2.5 Costos asimétricos de error
No todos los errores tienen el mismo impacto:
- **Falso negativo** (no detectar fraude): pérdida económica directa.
- **Falso positivo** (bloquear transacción legítima): mala experiencia del cliente.

Esto implica que la métrica de evaluación no puede ser solo accuracy.

---

## 3. ¿Por qué es un problema complejo de Machine Learning?

### 3.1 Métricas de evaluación no triviales
El accuracy es engañoso en datasets desbalanceados. Se requieren métricas como:
- Recall (sensibilidad)
- Precision
- F1-score
- ROC-AUC
- PR-AUC

Elegir la métrica adecuada depende del impacto de negocio.

---

### 3.2 Necesidad de técnicas avanzadas
Para abordar el problema se requieren técnicas como:
- Re-muestreo (SMOTE, undersampling, oversampling)
- Algoritmos sensibles al desbalance (class weights)
- Detección de anomalías
- Ensambles de modelos

---

### 3.3 Requerimientos de tiempo real
En producción, el modelo debe:
- Evaluar transacciones en milisegundos.
- Escalar a millones de eventos diarios.
- Mantener alta disponibilidad y baja latencia.

Esto añade complejidad a la arquitectura del sistema.

---

### 3.4 Evolución continua del fraude
El fraude no es estático:
- Los atacantes adaptan sus estrategias.
- Se requiere reentrenamiento frecuente.
- Es necesario monitorear el desempeño del modelo en producción.

---

### 3.5 Balance entre precisión y experiencia del cliente
El modelo debe encontrar un **equilibrio óptimo** entre:
- Detectar la mayor cantidad de fraudes posibles.
- Minimizar bloqueos injustificados a clientes legítimos.

Este balance convierte el problema en una combinación de **Machine Learning + optimización de negocio**.

---

## 4. Conclusión

La detección de fraude con tarjetas de crédito es un problema de Machine Learning **altamente complejo** debido al desbalance extremo de datos, la naturaleza dinámica del fraude, los altos costos de error y los estrictos requisitos de interpretabilidad y tiempo real. No se trata solo de entrenar un modelo preciso, sino de diseñar una solución robusta, explicable y alineada con los objetivos del negocio financiero.


# Entorno Virtual en Python (Ciencia de Datos)

Un **entorno virtual en Python** es un espacio aislado que permite instalar y administrar librerías de forma independiente para cada proyecto. En **ciencia de datos**, su uso es clave para evitar conflictos de versiones, garantizar reproducibilidad de los análisis y mantener proyectos organizados.

## Pasos básicos para usar un entorno virtual

1. **Crear el entorno**

python -m venv venv

![Diagrama de entornos virtuales en Python](https://jarroba.com/wp-content/uploads/2020/09/Crear-Virtualenv-entornos-virutals-en-Python-www.Jarroba.com_-1024x588.png)

# Random Forest para Clasificación

## Definición

**Random Forest (Bosque Aleatorio)** es un algoritmo de *Machine Learning* supervisado, utilizado para **clasificación y regresión**, que se basa en la construcción de múltiples **árboles de decisión** durante el entrenamiento.  

En clasificación, el modelo toma la **clase más votada** por todos los árboles como predicción final.

Es un modelo de tipo **ensemble**, específicamente **bagging (Bootstrap Aggregating)**, cuyo objetivo principal es **reducir el sobreajuste** y mejorar la capacidad de generalización.

---

## ¿Cómo funciona Random Forest en clasificación?

**1. Bootstrap del dataset**  
- Se generan múltiples subconjuntos del dataset original mediante muestreo aleatorio **con reemplazo**.  
- Cada subconjunto se utiliza para entrenar un árbol de decisión distinto.

**2. Selección aleatoria de variables (feature randomness)**  
- En cada nodo del árbol, solo se evalúa un subconjunto aleatorio de características.  
- Esto reduce la correlación entre los árboles y aumenta la diversidad del bosque.

**3. Entrenamiento de múltiples árboles**  
- Cada árbol se entrena de forma independiente.  
- Generalmente los árboles crecen sin poda, lo que aumenta su varianza individual.

**4. Votación mayoritaria**  
- Para una nueva observación, cada árbol predice una clase.  
- La clase final es la que obtiene **mayor número de votos**.

---

## Ventajas principales

- Reduce significativamente el **overfitting** frente a un árbol de decisión individual  
- Maneja bien relaciones **no lineales y complejas**  
- Es robusto ante **ruido y outliers**  
- Funciona correctamente con datasets de **alta dimensionalidad**  
- Requiere poco preprocesamiento de los datos  
- Puede manejar datasets **desbalanceados** con técnicas como `class_weight`  

---

## Conclusión

Random Forest es un algoritmo **robusto, estable y ampliamente utilizado** para clasificación.  
Es especialmente adecuado para problemas complejos como la **detección de fraude con tarjetas de crédito**, ya que combina múltiples modelos para producir predicciones más confiables mediante votación mayoritaria, reduciendo la varianza y mejorando la capacidad de generalización.

## XGBoost: funcionamiento y por qué fue el mejor modelo

**XGBoost (Extreme Gradient Boosting)** es un algoritmo de *Machine Learning* basado en **gradient boosting**, que construye árboles de decisión de manera **secuencial**. Cada nuevo árbol se entrena para **corregir los errores** de los árboles anteriores, optimizando directamente una función de pérdida y sumando las predicciones de todos los árboles para obtener el resultado final.

Funciona mediante:
- Optimización iterativa de una **función objetivo** (pérdida + regularización)
- Uso de **gradientes de primer y segundo orden** para mejorar la precisión del ajuste
- Construcción eficiente de árboles priorizando las divisiones con mayor ganancia
- Control de la complejidad del modelo mediante **regularización**

XGBoost fue el mejor modelo porque:
- Captura **relaciones no lineales complejas**
- Maneja eficazmente el **desbalance de clases**
- Reduce el **sobreajuste**
- Mostró **alta consistencia entre validación y prueba**
- Alcanzó el **AUC más alto** en el problema de detección de fraude

## tabla comparativa Random Forest vs XGBoost  

| Característica                     | Random Forest                                   | XGBoost                                              |
|-----------------------------------|-------------------------------------------------|------------------------------------------------------|
| Tipo de algoritmo                 | Ensemble basado en *bagging*                    | Ensemble basado en *gradient boosting*               |
| Forma de entrenamiento            | Árboles entrenados en paralelo                 | Árboles entrenados de forma secuencial               |
| Objetivo principal                | Reducir la varianza                             | Reduccir el sesgo y la varianza                      |
| Corrección de errores             | No corrige errores previos                      | Cada árbol corrige los errores del anterior          |
| Manejo de relaciones no lineales  | Bueno                                           | Excelente                                            |
| Regularización                    | Implícita (por promedio de árboles)             | Explícita (L1 y L2)                                  |
| Riesgo de overfitting             | Bajo                                            | Bajo si está bien regularizado                       |
| Manejo de datos desbalanceados    | Bueno (con ajustes)                             | Muy bueno (scale_pos_weight)                         |
| Uso de gradientes                 | No                                              | Sí (primer y segundo orden)                          |
| Velocidad de entrenamiento        | Rápida                                          | Media (más costosa computacionalmente)               |
| Interpretabilidad                 | Media                                           | Baja                                                 |
| Sensibilidad a hiperparámetros    | Baja                                            | Alta                                                 |
| Rendimiento típico (AUC)          | Bueno                                           | Muy alto                                             |
| Casos de uso ideales              | Modelos base y benchmarks rápidos               | Problemas complejos como detección de fraude         |
| Resultado en este dataset         | AUC ≈ 0.85                                      | AUC ≈ 0.97                                           |

