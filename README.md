#include <stdio.h>
#include <math.h>

// Declarar el dataset
double x[] = {5, 10, 15, 20, 25};  // Valores de "Batch Size"
double y[] = {10, 20, 30, 40, 50}; // Valores de "Output" correspondientes
int n = sizeof(x) / sizeof(x[0]);

// Función para calcular la regresión lineal
void calcular_regresion(double x[], double y[], int n, double *beta_0, double *beta_1) {
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    
    for (int i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
    }
    
    *beta_1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    *beta_0 = (sum_y - (*beta_1) * sum_x) / n;
}

// Función para calcular el coeficiente de correlación y el coeficiente de determinación
void calcular_correlacion_determinacion(double x[], double y[], int n, double beta_0, double beta_1, double *correlacion, double *r2) {
    double sum_y = 0, sum_y2 = 0, sum_ypred = 0, sum_ypred2 = 0, sum_yy_pred = 0;
    double y_mean = 0;

    // Calcular la media de y
    for (int i = 0; i < n; i++) {
        y_mean += y[i];
    }
    y_mean /= n;

    for (int i = 0; i < n; i++) {
        double y_pred = beta_0 + beta_1 * x[i];
        sum_y += (y[i] - y_mean) * (y[i] - y_mean);
        sum_ypred += (y_pred - y_mean) * (y_pred - y_mean);
        sum_yy_pred += (y[i] - y_mean) * (y_pred - y_mean);
    }

    *correlacion = sum_yy_pred / sqrt(sum_y * sum_ypred);
    *r2 = (*correlacion) * (*correlacion);
}

int main() {
    double beta_0, beta_1;
    calcular_regresion(x, y, n, &beta_0, &beta_1);
    
    // Imprimir la ecuación de la curva de regresión
    printf("Curva de regresión: Output = %.2lf + %.2lf * Batch_Size\n", beta_0, beta_1);

    // Realizar cinco predicciones (hardcoded)
    double batch_sizes[] = {5, 10, 15, 30, 35};  // Valores conocidos y desconocidos
    for (int i = 0; i < 5; i++) {
        double y_pred = beta_0 + beta_1 * batch_sizes[i];
        printf("Para Batch_Size = %.2lf, Output predicho = %.2lf\n", batch_sizes[i], y_pred);
    }

    // Calcular y mostrar correlación y coeficiente de determinación
    double correlacion, r2;
    calcular_correlacion_determinacion(x, y, n, beta_0, beta_1, &correlacion, &r2);
    printf("Coeficiente de correlación = %.2lf\n", correlacion);
    printf("Coeficiente de determinación (R^2) = %.2lf\n", r2);

    return 0;
}
