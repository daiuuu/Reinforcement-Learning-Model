# Reinforcement Learning Model

## Descripción

Este proyecto implementa un modelo de **Aprendizaje por Refuerzo (Reinforcement Learning)** utilizando el algoritmo **Q-Learning** sobre el entorno **FrozenLake-v1** de Gymnasium.

El objetivo del agente es aprender, mediante prueba y error, a desplazarse por un lago congelado desde la posición inicial hasta la meta, evitando caer en los agujeros del mapa.

Durante el entrenamiento el agente no conoce el camino correcto. A medida que interactúa con el entorno, recibe recompensas y actualiza una **Tabla Q**, que almacena el valor esperado de realizar cada acción en cada estado.

Una vez finalizado el entrenamiento, la Tabla Q se guarda en un archivo para reutilizarla posteriormente sin necesidad de volver a entrenar el modelo.

---

## Tecnologías utilizadas

- Python 3
- NumPy
- Gymnasium (FrozenLake-v1)

---

## Estructura del proyecto

```
.
├── rl_model.py
├── q_table.npy
├── README.md
└── requirements.txt
```

---

## Instalación

Clonar el repositorio:

```bash
git clone <url-del-repositorio>
cd <nombre-del-repositorio>
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

---

## Dependencias

```
numpy
gymnasium
```

También pueden instalarse manualmente:

```bash
pip install numpy gymnasium
```

---

## Funcionamiento

El entrenamiento sigue el algoritmo clásico de **Q-Learning**.

Durante cada episodio:

1. El entorno se reinicia.
2. El agente observa el estado actual.
3. Selecciona una acción utilizando una política **ε-greedy**.
4. Ejecuta la acción.
5. Recibe una recompensa.
6. Actualiza la Tabla Q.
7. Continúa hasta finalizar el episodio.

Con el paso de los episodios, el valor de **epsilon** disminuye, permitiendo que el agente explore mucho al principio y aproveche cada vez más el conocimiento adquirido.

---

## Parámetros del modelo

| Parámetro | Valor | Descripción |
|-----------|------:|-------------|
| Learning Rate | 0.8 | Velocidad de aprendizaje |
| Gamma | 0.95 | Factor de descuento para recompensas futuras |
| Epsilon inicial | 1.0 | Probabilidad inicial de exploración |
| Epsilon mínimo | 0.01 | Exploración mínima |
| Decaimiento de epsilon | 0.999 | Reducción gradual de la exploración |
| Episodios | 10000 | Cantidad de entrenamientos |

---

## Entrenamiento

Para entrenar el modelo ejecutar:

```bash
python rl_model.py
```

Durante el entrenamiento se mostrará el progreso cada 200 episodios indicando la tasa de victorias alcanzada.

Al finalizar se generará automáticamente el archivo:

```
q_table.npy
```

que contiene la Tabla Q aprendida.

---

## Evaluación

Una vez entrenado el modelo, se ejecuta una evaluación sobre 100 episodios.

Durante esta etapa:

- se carga la Tabla Q previamente guardada;
- el agente deja de explorar (`epsilon = 0`);
- únicamente utiliza el conocimiento adquirido para tomar decisiones.

Finalmente se informa el porcentaje de victorias obtenido.

Ejemplo:

```
Evaluación sobre 100 episodios | Tasa de victorias: 97.0%
```

---

## Algoritmo utilizado

El proyecto implementa **Q-Learning**, un algoritmo de aprendizaje por refuerzo basado en diferencias temporales (Temporal Difference Learning).

La actualización de la Tabla Q se realiza mediante la ecuación:

\[
Q(s,a)=Q(s,a)+\alpha\left[r+\gamma\max Q(s',a')-Q(s,a)\right]
\]

donde:

- **Q(s,a)** representa el valor estimado de realizar una acción en un estado.
- **α** es la tasa de aprendizaje.
- **γ** es el factor de descuento.
- **r** es la recompensa obtenida.
- **s'** representa el siguiente estado.

---

## Objetivo del proyecto

Este proyecto tiene como finalidad comprender los conceptos fundamentales del Aprendizaje por Refuerzo, implementando desde cero un agente capaz de aprender una política óptima mediante el algoritmo Q-Learning, sin utilizar redes neuronales ni modelos preentrenados.

---

## Autor

**Daiana Soria Piola**
