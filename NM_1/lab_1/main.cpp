#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib> 
#include <ctime>   
#include <iomanip> 
#include <chrono>  

using matrix = std::vector<std::vector<double>>;
using vector = std::vector<double>;

// Вывести первые count координат
void print_vector_head(const vector& vec, int count, const std::string& title) {
    std::cout << title << " (первые " << count << " координат):" << std::endl;
    for (int i = 0; i < count && i < vec.size(); ++i) {
        std::cout << "x[" << i << "] = " << vec[i] << std::endl;
    }
    std::cout << std::endl;
}

// Функция для вычисления нормы-бесконечность
double infinity_norm(const vector& vec) {
    double max_val = 0.0;
    for (double val : vec) {
        if (std::abs(val) > max_val) {
            max_val = std::abs(val);
        }
    }
    return max_val;
}

// Метод Гаусса без выбора ведущего элемента
vector solve_gauss_without_chose(matrix A, vector b) {
    int n = A.size();

    
    for (int k = 0; k < n - 1; ++k) {
        if (std::abs(A[k][k]) < 1e-12) { 
            std::cerr << "Ошибка: Нулевой главный элемент на шаге " << k << ". Метод без выбора элемента неприменим." << std::endl;
            return {};
        }

        for (int i = k + 1; i < n; ++i) {
            double lik = A[i][k] / A[k][k];
            
            for (int j = k; j < n; ++j) {
                A[i][j] -= lik * A[k][j];
            }

            b[i] -= lik * b[k];
        }
    }

    vector x(n);
    double sum;
    for (int i = n - 1; i >= 0; --i) {
        sum = 0;
        for (int j = i + 1; j < n; ++j) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
    }

    return x;
}

// Метод Гаусса с выбором ведущего элемента по столбцу
vector solve_gauss_with_chose(matrix A, vector b) {
    int n = A.size();

    for (int k = 0; k < n - 1; ++k) {
        int pivot_row = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(A[i][k]) > std::abs(A[pivot_row][k])) {
                pivot_row = i;
            }
        }

        if (pivot_row != k) {
            std::swap(A[k], A[pivot_row]);
            std::swap(b[k], b[pivot_row]);
        }

        if (std::abs(A[k][k]) < 1e-12) {
             std::cerr << "Ошибка: Матрица вырождена. Решения нет." << std::endl;
             return {};
        }

        for (int i = k + 1; i < n; ++i) {
            double lik = A[i][k] / A[k][k];
            for (int j = k; j < n; ++j) {
                A[i][j] -= lik * A[k][j];
            }
            b[i] -= lik * b[k];
        }
    }

    vector x(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0;
        for (int j = i + 1; j < n; ++j) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
    }

    return x;
}


int main() {
    setlocale(LC_ALL, "Russian");
    srand(time(0));

    const int n = 2000;
    const int m = 15;

    std::cout << "Постановка задачи:" << std::endl;
    std::cout << "n = " << n << ", m = " << m << std::endl << std::endl;

    matrix A(n, vector(n));
    vector x_exact(n);
    vector b(n, 0.0);

    for (int i = 0; i < n; ++i) {
        x_exact[i] = m + i;
        for (int j = 0; j < n; ++j) {
            A[i][j] = -100.0 + 200.0 * rand() / RAND_MAX;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            b[i] += A[i][j] * x_exact[j];
        }
    }

    std::cout << "--- 1. Метод Гаусса БЕЗ выбора ведущего элемента ---" << std::endl;
    auto start_no_chose = std::chrono::high_resolution_clock::now();
    vector x_no_chose = solve_gauss_without_chose(A, b);
    auto end_no_chose = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_no_chose = end_no_chose - start_no_chose;

    if (!x_no_chose.empty()) {
        // Вычисление невязки r = b - A*x*
        vector residual_no_chose(n);
        for (int i = 0; i < n; ++i) {
            double ax_sum = 0;
            for (int j = 0; j < n; ++j) {
                ax_sum += A[i][j] * x_no_chose[j];
            }
            residual_no_chose[i] = b[i] - ax_sum;
        }

        // Вычисление погрешности diff = x* - x
        vector diff_no_pivoting(n);
        for (int i = 0; i < n; ++i) {
            diff_no_pivoting[i] = x_no_chose[i] - x_exact[i];
        }

        print_vector_head(x_no_chose, 5, "Приближенное решение x*");
        std::cout << std::scientific << std::setprecision(5);
        std::cout << "Норма вектора невязки ||Ax* - b||: " << infinity_norm(residual_no_chose) << std::endl;
        std::cout << "Относительная погрешность ||x* - x|| / ||x||: " << infinity_norm(diff_no_pivoting) / infinity_norm(x_exact) << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Время выполнения: " << duration_no_chose.count() << " секунд" << std::endl << std::endl;
    }

    // --- Решение и анализ для метода С выбором элемента ---
    std::cout << "--- 2. Метод Гаусса С выбором ведущего элемента по столбцу ---" << std::endl;
    auto start_with_chose = std::chrono::high_resolution_clock::now();
    vector x_with_chose = solve_gauss_with_chose(A, b);
    auto end_with_chose = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_with_chose = end_with_chose - start_with_chose;

    if (!x_with_chose.empty()) {
        // Вычисление невязки r = b - A*x*
        vector residual_with_chose(n);
        for (int i = 0; i < n; ++i) {
            double ax_sum = 0;
            for (int j = 0; j < n; ++j) {
                ax_sum += A[i][j] * x_with_chose[j];
            }
            residual_with_chose[i] = b[i] - ax_sum;
        }

        // Вычисление погрешности diff = x* - x
        vector diff_with_chose(n);
        for (int i = 0; i < n; ++i) {
            diff_with_chose[i] = x_with_chose[i] - x_exact[i];
        }

        print_vector_head(x_with_chose, 5, "Приближенное решение x*");
        std::cout << std::scientific << std::setprecision(5);
        std::cout << "Норма вектора невязки ||Ax* - b||: " << infinity_norm(residual_with_chose) << std::endl;
        std::cout << "Относительная погрешность ||x* - x|| / ||x||: " << infinity_norm(diff_with_chose) / infinity_norm(x_exact) << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Время выполнения: " << duration_with_chose.count() << " секунд" << std::endl;
    }

    return 0;
}