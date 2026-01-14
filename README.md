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
Para clasificación, el modelo toma la **clase más votada** por todos los árboles como resultado final.

Es un modelo de tipo **ensemble**, específicamente *bagging* (Bootstrap Aggregating), cuyo objetivo principal es **reducir el sobreajuste** y mejorar la capacidad de generalización.

---

## ¿Cómo funciona Random Forest en clasificación?

1. **Bootstrap del dataset**  
   - Se crean múltiples subconjuntos del dataset original mediante muestreo aleatorio **con reemplazo**.
   - Cada subconjunto se utiliza para entrenar un árbol de decisión distinto.

2. **Selección aleatoria de variables (feature randomness)**  
   - En cada nodo del árbol, solo se evalúa un subconjunto aleatorio de características.
   - Esto reduce la correlación entre árboles y aumenta la diversidad del bosque.

3. **Entrenamiento de múltiples árboles**  
   - Cada árbol se entrena de forma independiente.
   - Los árboles suelen crecer sin poda (árboles profundos).

4. **Votación mayoritaria**  
   - Para una nueva observación, cada árbol predice una clase.
   - La clase final es la que obtiene **mayor número de votos**.

---

## Representación matemática (fórmula)

Sea:
- \( T = \{h_1(x), h_2(x), \dots, h_N(x)\} \) el conjunto de árboles entrenados.
- \( h_i(x) \) la predicción del árbol \( i \).
- \( N \) el número total de árboles.

La predicción final del Random Forest para clasificación es:

\[
\hat{y} = \underset{c \in C}{\arg\max} \sum_{i=1}^{N} \mathbb{I}(h_i(x) = c)
\]

Donde:
- \( C \) es el conjunto de clases.
- \( \mathbb{I}(\cdot) \) es la función indicadora, que vale 1 si la condición se cumple y 0 en caso contrario.

---

## Función de impureza (criterio de división)

En clasificación, Random Forest suele utilizar:

### Índice de Gini
\[
Gini = 1 - \sum_{j=1}^{K} p_j^2
\]

### Entropía
\[
Entropy = - \sum_{j=1}^{K} p_j \log_2(p_j)
\]

Donde:
- \( K \) es el número de clases.
- \( p_j \) es la proporción de muestras de la clase \( j \) en el nodo.

---

## Ventajas principales

- Maneja bien **datasets desbalanceados** (con ajustes).
- Reduce el **overfitting** frente a un árbol individual.
- Funciona bien con datos no lineales.
- Requiere poco preprocesamiento.

---

## Conclusión

Random Forest es un algoritmo robusto y ampliamente utilizado en clasificación, especialmente en problemas complejos como **detección de fraude**, debido a su capacidad para combinar múltiples modelos débiles y producir predicciones más estables y precisas mediante votación mayoritaria.

