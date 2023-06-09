#include <stdio.h>
#include <time.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

void printMatrix(gsl_matrix* matrix) {
    for (size_t i = 0; i < matrix->size1; i++) {
        for (size_t j = 0; j < matrix->size2; j++) {
            printf("%g ", gsl_matrix_get(matrix, i, j));
        }
        printf("\n");
    }
}

int main() {
    struct timespec start_time, end_time;
    double execution_time;

    // Set the size of the matrix
    int size = 10;
    int index_i , index_j;

    while (scanf("%d", &size)!=EOF){

    clock_gettime(CLOCK_MONOTONIC, &start_time); // Obtém o tempo de início
    double value;//deletar

    // Allocate memory for the random matrix
    gsl_matrix* matrix = gsl_matrix_alloc(size, size);

    // Fill the matrix with random values
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            scanf("%lf", &value);
            gsl_matrix_set(matrix, i, j, value);
            
            //gsl_matrix_set(matrix, i, j, rand() % 100 + 1);
        }
    }

    scanf("%lf", &value);
    for (size_t i = 0; i < size; i++)
        for (size_t j = 0; j < size; j++)
            scanf("%lf", &value);
    scanf("%d", &index_i);
    scanf("%d", &index_j);
    scanf("%lf", &value);

    printf("Matriz A:\n");printMatrix(matrix);

    // Create a copy of the matrix to store the inverse
    gsl_matrix* inverse = gsl_matrix_alloc(size, size);
    gsl_matrix_memcpy(inverse, matrix);

    // Perform the LU decomposition to calculate the inverse
    gsl_permutation* perm = gsl_permutation_alloc(size);
    int signum;
    gsl_linalg_LU_decomp(inverse, perm, &signum);
    gsl_linalg_LU_invert(inverse, perm, inverse);

    printf("\nMatriz A Inversa:\n");printMatrix(inverse);

    // Free allocated memory
    gsl_matrix_free(matrix);
    gsl_matrix_free(inverse);
    gsl_permutation_free(perm);

    clock_gettime(CLOCK_MONOTONIC, &end_time); // Obtém o tempo de término
    execution_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0; // Diferença de segundos em milissegundos
    execution_time += (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0; // Diferença de nanossegundos em milissegundos
    printf("%f\n", execution_time);
    }
    return 0;
}
