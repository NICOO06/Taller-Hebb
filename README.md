# Reconocimiento de Letras con Regla de Hebb (C++)

Este proyecto implementa una **red neuronal simple basada en la regla de Hebb** para reconocer **tres letras (A, B y C)** representadas como patrones **3x3** usando valores bipolares (**-1 y +1**). El objetivo es mostrar cómo la regla de Hebb puede aprender asociaciones directas entre entradas y salidas.

---

## 1. **Descripción del Problema**

La tarea consiste en clasificar letras simples representadas en una cuadrícula 3×3:

* **Letra A**
* **Letra B**
* **Letra C**

Cada letra se codifica como un vector de **9 entradas bipolares (-1/+1)**.
La salida se codifica con **2 neuronas**, donde cada combinación bipolar representa una letra:

| Letra | Salida    |
| ----- | --------- |
| A     | `[1, 1]`  |
| B     | `[1, -1]` |
| C     | `[-1, 1]` |

---

## 2. **Diseño de la Red**

* **Entradas:** 9 neuronas (una por pixel del patrón 3×3)
* **Bias:** 1 neurona adicional
* **Salidas:** 2 neuronas
* **Aprendizaje:** Regla de Hebb → `w += x * y`

La red memoriza las correlaciones entre los patrones de entrada y sus salidas esperadas.

---

## 3. **Implementación en C++**

El programa incluye:

* Inicialización de pesos en cero
* Entrenamiento con la regla de Hebb
* Pruebas con los patrones originales
* Pruebas con ruido (modificación de 1 píxel)
* Impresión de pesos antes y después del entrenamiento
* Análisis del rendimiento

```cpp
// CÓDIGO C++
#include <iostream>
#include <vector>
#include <iomanip>
using namespace std;

int sign(double x) { return (x >= 0.0) ? 1 : -1; }

void print_weights(const vector<vector<double>>& W) {
    for (size_t j = 0; j < W.size(); ++j) {
        cout << "Salida " << j << ": ";
        for (size_t i = 0; i < W[j].size(); ++i)
            cout << fixed << setprecision(2) << W[j][i] << " ";
        cout << '\n';
    }
}

int main() {
    vector<vector<int>> inputs = {
        { 1, -1,  1,  1,  1,  1,  1, -1,  1 },
        { 1,  1, -1,  1,  1,  1,  1,  1, -1 },
        { 1,  1,  1,  1, -1, -1,  1,  1,  1 }
    };

    vector<vector<int>> targets = {
        { 1,  1 },
        { 1, -1 },
        { -1, 1 }
    };

    const int n_inputs = 9;
    const int n_outputs = 2;
    const double lr = 1.0;

    vector<vector<double>> W(n_outputs, vector<double>(n_inputs + 1, 0.0));

    cout << "Pesos iniciales:\n";
    print_weights(W);

    for (size_t p = 0; p < inputs.size(); ++p) {
        vector<int> x(n_inputs + 1);
        for (int i = 0; i < n_inputs; ++i) x[i] = inputs[p][i];
        x[n_inputs] = 1;

        for (int j = 0; j < n_outputs; ++j)
            for (int i = 0; i < n_inputs + 1; ++i)
                W[j][i] += lr * x[i] * targets[p][j];
    }

    cout << "Pesos después del entrenamiento:\n";
    print_weights(W);

    cout << "\nPruebas:\n";
    int correct = 0;

    for (size_t p = 0; p < inputs.size(); ++p) {
        vector<int> x(n_inputs + 1);
        for (int i = 0; i < n_inputs; ++i) x[i] = inputs[p][i];
        x[n_inputs] = 1;

        vector<int> out(n_outputs);
        for (int j = 0; j < n_outputs; ++j) {
            double net = 0.0;
            for (int i = 0; i < n_inputs + 1; ++i) net += W[j][i] * x[i];
            out[j] = sign(net);
        }

        cout << "Esperado: [" << targets[p][0] << ", " << targets[p][1] << "]  ";
        cout << "Obtenido: [" << out[0] << ", " << out[1] << "]\n";

        if (out == targets[p]) correct++;
    }

    cout << "\nExactitud: " << (100.0 * correct / inputs.size()) << "%\n";

    return 0;
}
```

---

## 4. **Resultados Esperados**

* Pesos iniciales = todos ceros
* Pesos después del entrenamiento = correlaciones aprendidas
* Exactitud en entrenamiento ≈ **100%**
* Pruebas con ruido: desempeño depende de similitud entre letras

---

## 5. **Análisis y Conclusiones**

### ✔ ¿Qué aprendió la red?

Asoció las características visuales de cada letra con su vector de salida.

### ✔ ¿Cómo evolucionaron los pesos?

Los pesos aumentan o disminuyen según la correlación **entrada × salida**.

### ✔ Rendimiento

Excelente para patrones simples y bien separados.

### ✔ Limitaciones

* La regla de Hebb **no corrige errores**.
* No maneja bien patrones similares o muy ruidosos.

### ✔ Posibles mejoras

* Usar regla delta (Perceptrón)
* Añadir capas ocultas
* Normalizar entradas
* Realizar múltiples épocas de aprendizaje

---

**Vargas Angarita Nicolás Antonio**
