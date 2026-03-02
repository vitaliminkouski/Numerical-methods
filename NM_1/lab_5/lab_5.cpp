#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <numeric>

// Псевдонимы типов
using matrix = std::vector<std::vector<double>>;
using vec = std::vector<double>;

// --- Вспомогательные функции (векторные операции) ---

// Скалярное произведение векторов (алгоритм sxz )
double dot_product(const vec& a, const vec& b) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Умножение матрицы на вектор (алгоритм mvAx )
vec mat_vec_mult(const matrix& A, const vec& x) {
    size_t n = A.size();
    vec result(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}

// Евклидова норма вектора
double vector_norm(const vec& v) {
    return std::sqrt(dot_product(v, v));
}

// Разность векторов: a - b
vec vec_diff(const vec& a, const vec& b) {
    vec res(a.size());
    for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] - b[i];
    return res;
}

// Сумма векторов: a + b
vec vec_add(const vec& a, const vec& b) {
    vec res(a.size());
    for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] + b[i];
    return res;
}

// Умножение вектора на скаляр: scalar * v
vec vec_scale(double scalar, const vec& v) {
    vec res(v.size());
    for (size_t i = 0; i < v.size(); ++i) res[i] = scalar * v[i];
    return res;
}

// Вывод первых элементов
void print_vector_head(const vec& v, const std::string& title, int count) {
    std::cout << title << std::endl;
    for (int i = 0; i < count && i < v.size(); ++i) {
        std::cout << "  [" << i << "] = " << v[i] << std::endl;
    }
}

// Вычисление нормы невязки ||f - Ax||
double calc_residual_norm(const matrix& A, const vec& x, const vec& f) {
    vec Ax = mat_vec_mult(A, x);
    vec r = vec_diff(f, Ax); // r = f - Ax (в методичке r = f - Ax*) [cite: 3]
    return vector_norm(r);
}

// --- Метод 1: LDLT Разложение (из Лаб 2) ---
vec solve_ldlt(matrix A, const vec& b) {
    int n = A.size();
    vec t(n);
    // Факторизация
    for (int k = 0; k < n - 1; ++k) {
        double d_kk = A[k][k];
        for (int i = k + 1; i < n; ++i) {
            t[i] = A[i][k];
            A[i][k] /= d_kk;
        }
        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j <= i; ++j) {
                 A[i][j] -= A[i][k] * t[j];
            }
        }
    }
    // Решение Ly = b
    vec y(n);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < i; ++j) sum += A[i][j] * y[j];
        y[i] = b[i] - sum;
    }
    // Решение Dz = y
    vec z(n);
    for (int i = 0; i < n; ++i) z[i] = y[i] / A[i][i];

    // Решение L^T x = z
    vec x(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) sum += A[j][i] * x[j];
        x[i] = z[i] - sum;
    }
    return x;
}

// --- Метод 2: Сопряженные градиенты (CG) ---
// Алгоритм основан на псевдокоде 
vec solve_cg(const matrix& A, const vec& f, double epsilon, int max_iter, int& iterations_out) {
    int n = f.size();
    vec x(n, 0.0); // x0 = 0 (начальное приближение) 
    
    // r0 = f - A*x0 = f (так как x0=0) 
    vec r = f; 
    vec p = r; // p0 = r0 
    
    double r_norm = vector_norm(r);
    // (r_j, r_j) - скалярное произведение невязки на саму себя
    double rr_curr = dot_product(r, r); 
    
    iterations_out = 0;

    for (int j = 0; j < max_iter; ++j) {
        iterations_out++;
        
        // Проверка критерия остановки ||r|| < epsilon 
        if (std::sqrt(rr_curr) < epsilon) {
            break;
        }

        vec Ap = mat_vec_mult(A, p); // Ap_j
        double Ap_p = dot_product(Ap, p); // (Ap_j, p_j)
        
        // alpha_j = (r_j, r_j) / (Ap_j, p_j) 
        double alpha = rr_curr / Ap_p; 
        
        // x_{j+1} = x_j + alpha * p_j 
        x = vec_add(x, vec_scale(alpha, p));
        
        // r_{j+1} = r_j - alpha * Ap_j 
        r = vec_diff(r, vec_scale(alpha, Ap));
        
        double rr_next = dot_product(r, r); // (r_{j+1}, r_{j+1})
        
        // beta_j = (r_{j+1}, r_{j+1}) / (r_j, r_j) [cite: 48]
        double beta = rr_next / rr_curr;
        
        // p_{j+1} = r_{j+1} + beta * p_j [cite: 49]
        p = vec_add(r, vec_scale(beta, p));
        
        rr_curr = rr_next; // Обновляем для следующего шага
    }
    
    if (iterations_out >= max_iter) {
        std::cout << "Внимание: Достигнуто максимальное число итераций (" << max_iter << ")!" << std::endl;
    }
    
    return x;
}

int main() {
    // Параметры задания 
    const int n = 2000;
    const int m_student = 15; 
    const int k_group = 4;  
    const int max_iter = 2000; // l_max 

    std::cout << "--- Лабораторная работа №5: Метод сопряженных градиентов ---\n";
    std::cout << "Входные данные: n = " << n << ", m = " << m_student << ", k = " << k_group << "\n\n";

    // 1. Генерация матрицы A и точного решения ==
    matrix A(n, vec(n));
    vec x_exact(n);
    vec f(n, 0.0); // Правая часть (обозначена f в лаб 5) 

    std::mt19937 gen(123); 
    std::uniform_real_distribution<> dis(-100.0, 0.0);

    // Заполнение недиагональных элементов
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            A[i][j] = dis(gen);
            A[j][i] = A[i][j]; // Симметричность
        }
    }

    // Заполнение диагональных элементов (обеспечение положительной определенности)
    double sum_first_row = 0.0;
    for (int j = 1; j < n; ++j) sum_first_row += A[0][j];
    A[0][0] = -sum_first_row + 10 * k_group;

    for (int i = 1; i < n; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            row_sum += A[i][j];
        }
        A[i][i] = -row_sum + 10 * k_group;
    }

    // Формирование правой части f = A * x_exact
    for (int i = 0; i < n; ++i) x_exact[i] = m_student + i;
    f = mat_vec_mult(A, x_exact);

    std::cout << std::fixed << std::setprecision(8);

    // --- ЭТАП 1: Решение методом LDLT ---
    std::cout << "=== 1. Метод LDLT (прямой) ===\n";
    vec x_ldlt;
    double time_ldlt;
    double residual_norm_ldlt;
    
    {
        auto start = std::chrono::high_resolution_clock::now();
        x_ldlt = solve_ldlt(A, f); // Передаем копию A, так как solve_ldlt её меняет
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        time_ldlt = diff.count();
        
        residual_norm_ldlt = calc_residual_norm(A, x_ldlt, f);
        
        print_vector_head(x_ldlt, "Первые 5 координат x*:", 5);
        std::cout << "Норма невязки ||f - Ax*||: " << residual_norm_ldlt << std::endl;
        
        vec diff_vec = vec_diff(x_ldlt, x_exact);
        double rel_error = vector_norm(diff_vec) / vector_norm(x_exact);
        std::cout << "Относительная погрешность: " << rel_error << std::endl;
        std::cout << "Время выполнения: " << time_ldlt << " с\n\n";
    }

    // --- ЭТАП 2: Решение методом CG (Сопряженных Градиентов) ---
    std::cout << "=== 2. Метод CG (итерационный) ===\n";
    std::cout << "Целевая точность (epsilon) взята из LDLT: " << residual_norm_ldlt << std::endl;

    vec x_cg;
    double time_cg;
    int iterations;
    
    {
        auto start = std::chrono::high_resolution_clock::now();
        // Используем residual_norm_ldlt как epsilon 
        x_cg = solve_cg(A, f, residual_norm_ldlt, max_iter, iterations);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        time_cg = diff.count();
        
        double residual_norm_cg = calc_residual_norm(A, x_cg, f);
        
        print_vector_head(x_cg, "Первые 5 координат x*:", 5);
        std::cout << "Количество итераций: " << iterations << std::endl;
        std::cout << "Норма невязки ||f - Ax*||: " << residual_norm_cg << std::endl;
        
        vec diff_vec = vec_diff(x_cg, x_exact);
        double rel_error = vector_norm(diff_vec) / vector_norm(x_exact);
        std::cout << "Относительная погрешность: " << rel_error << std::endl;
        std::cout << "Время выполнения: " << time_cg << " с\n";
    }
    
    std::cout << "\n--- Сравнение ---\n";
    std::cout << "Ускорение (LDLt / CG): " << time_ldlt / time_cg << " раз(а)\n";

    return 0;
}