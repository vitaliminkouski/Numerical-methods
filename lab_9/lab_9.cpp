#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>

using namespace std;

// === Константы и Типы ===
const int N = 4;
const float EPS = 1e-6; // Точность вычислений
typedef vector<vector<float>> Matrix;

// === Матричные операции ===

Matrix createIdentity(int n) {
    Matrix res(n, vector<float>(n, 0.0));
    for (int i = 0; i < n; i++) res[i][i] = 1.0;
    return res;
}

void printMatrix(const string& title, const Matrix& m) {
    cout << title << ":" << endl;
    for (const auto& row : m) {
        for (float val : row) {
            cout << setw(10) << fixed << setprecision(4) << val << " ";
        }
        cout << endl;
    }
    cout << endl;
}

Matrix multiply(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<float>(n, 0.0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

// Умножение матрицы на вектор
vector<float> multiplyMV(const Matrix& A, const vector<float>& v) {
    int n = A.size();
    vector<float> res(n, 0.0);
    for(int i=0; i<n; ++i)
        for(int j=0; j<n; ++j)
            res[i] += A[i][j] * v[j];
    return res;
}

// Метод Данилевского (возвращает true если успех)
bool danilevskyMethod(Matrix A, Matrix& Phi, vector<Matrix>& M_list, vector<float>& p) {
    int n = A.size();
    Phi = A;
    M_list.clear();

    for (int i = n - 1; i > 0; i--) {
        float pivot = Phi[i][i - 1];
        if (fabs(pivot) < 1e-8) return false; // Нерегулярный случай

        Matrix M = createIdentity(n);
        Matrix M_inv = createIdentity(n);

        M[i - 1][i - 1] = 1.0f / pivot;
        for (int j = 0; j < n; j++) {
            if (j != i - 1) M[i - 1][j] = -Phi[i][j] / pivot;
        }

        for (int j = 0; j < n; j++) M_inv[i - 1][j] = Phi[i][j];

        Phi = multiply(multiply(M_inv, Phi), M);
        M_list.push_back(M);
    }

    p.clear();
    for (int j = 0; j < n; j++) p.push_back(Phi[0][j]);
    return true;
}

// Работа с полиномами и численное решение ===

// Вычисление значения полинома P(x) = x^4 - p1*x^3 - p2*x^2 - p3*x - p4
// Коэффициенты p передаются как {p1, p2, p3, p4}
float valP(float x, const vector<float>& p) {
    return pow(x, 4) - p[0]*pow(x, 3) - p[1]*pow(x, 2) - p[2]*x - p[3];
}

// Первая производная P'(x) = 4x^3 - 3p1*x^2 - 2p2*x - p3
float valdP(float x, const vector<float>& p) {
    return 4*pow(x, 3) - 3*p[0]*pow(x, 2) - 2*p[1]*x - p[2];
}

// Вторая производная P''(x) = 12x^2 - 6p1*x - 2p2
float valddP(float x, const vector<float>& p) {
    return 12*pow(x, 2) - 6*p[0]*x - 2*p[1];
}

// Третья производная P'''(x) = 24x - 6p1 (для метода Ньютона при поиске корней P')
float valdddP(float x, const vector<float>& p) {
    return 24*x - 6*p[0];
}

// Метод деления отрезка пополам (Bisection)
// solveFunc - указатель на функцию, которую решаем
float bisectionMethod(float a, float b, float (*func)(float, const vector<float>&), const vector<float>& p) {
    if (func(a, p) * func(b, p) > 0) return NAN; // Корня нет или четное число
    float c;
    int iter = 0;
    while ((b - a) / 2 > EPS && iter < 1000) {
        c = (a + b) / 2;
        if (func(c, p) == 0.0) break;
        if (func(a, p) * func(c, p) < 0) b = c;
        else a = c;
        iter++;
    }
    return (a + b) / 2;
}

// Метод Ньютона
// func - функция, dfunc - её производная
float newtonMethod(float x0, float (*func)(float, const vector<float>&), float (*dfunc)(float, const vector<float>&), const vector<float>& p) {
    float x = x0;
    int iter = 0;
    while (iter < 100) {
        float fVal = func(x, p);
        float dfVal = dfunc(x, p);
        if (fabs(dfVal) < 1e-12) break; // Деление на ноль
        float x_new = x - fVal / dfVal;
        if (fabs(x_new - x) < EPS) return x_new;
        x = x_new;
        iter++;
    }
    return x;
}

int main() {
    setlocale(LC_ALL, "Russian");
    srand(static_cast<unsigned int>(time(0)));

    cout << "=== ЛР №9: Метод Ньютона для нелинейных уравнений ===" << endl;
    
    // 1. Подготовка данных
    Matrix A(N, vector<float>(N));
    Matrix Phi;
    vector<Matrix> M_list;
    vector<float> p; // Коэффициенты p1..p4
    bool success = false;

    // Генерируем, пока не получим регулярный случай
    do {
        for(int i=0; i<N; ++i)
            for(int j=0; j<N; ++j)
                A[i][j] = (rand()%21 - 10) + (float)rand()/RAND_MAX; // -10..10
        
        success = danilevskyMethod(A, Phi, M_list, p);
    } while (!success);

    printMatrix("Матрица A", A);
    cout << "Коэффициенты характеристического многочлена P(lambda):" << endl;
    for(size_t i=0; i<p.size(); ++i) cout << "p" << i+1 << " = " << p[i] << endl;
    cout << endl;

    // 2. Отделение корней
    //  Решаем P''(lambda) = 0 -> Квадратное уравнение 12x^2 - 6p1*x - 2p2 = 0
    // Сократим на 2: 6x^2 - 3p1*x - p2 = 0
    float a_sq = 6;
    float b_sq = -3 * p[0];
    float c_sq = -p[1];
    float D = b_sq*b_sq - 4*a_sq*c_sq;

    vector<float> ddp_roots;
    if (D >= 0) {
        ddp_roots.push_back((-b_sq - sqrt(D)) / (2*a_sq));
        ddp_roots.push_back((-b_sq + sqrt(D)) / (2*a_sq));
        sort(ddp_roots.begin(), ddp_roots.end());
    } else {
        cout << "P''(lambda) не имеет вещественных корней. P'(lambda) монотонна." << endl;
    }

    // Решаем P'(lambda) = 0
    // Интервалы поиска задаются корнями P'' и границами "бесконечности" (возьмем -100 и 100)
    vector<float> search_points = {-100.0f};
    search_points.insert(search_points.end(), ddp_roots.begin(), ddp_roots.end());
    search_points.push_back(100.0f);

    vector<float> dp_roots;
    cout << "--- Поиск корней производной P'(lambda) ---" << endl;
    for (size_t i = 0; i < search_points.size() - 1; ++i) {
        float left = search_points[i];
        float right = search_points[i+1];
        
        // Метод бисекции (для грубой оценки)
        float root_bisect = bisectionMethod(left, right, valdP, p);
        
        if (!isnan(root_bisect)) {
            // Уточнение методом Ньютона (решаем P'=0, производная P'')
            // Начальное приближение берем из бисекции
            float root_newton = newtonMethod(root_bisect, valdP, valddP, p);
            
            // Проверка, что корень не дублируется (для близких интервалов)
            bool exists = false;
            for(float r : dp_roots) if(fabs(r - root_newton) < 1e-4) exists = true;
            
            if(!exists) {
                dp_roots.push_back(root_newton);
                cout << "Найден корень P'(lambda) на [" << left << ", " << right << "]: " << root_newton << endl;
            }
        }
    }
    sort(dp_roots.begin(), dp_roots.end());

    // Решаем P(lambda) = 0 (Главная цель)
    vector<float> p_roots;
    vector<float> final_intervals = {-100.0f}; // Расширяем границы, если нужно
    final_intervals.insert(final_intervals.end(), dp_roots.begin(), dp_roots.end());
    final_intervals.push_back(100.0f);

    cout << "\n--- Поиск собственных чисел (корней P(lambda)) ---" << endl;
    for (size_t i = 0; i < final_intervals.size() - 1; ++i) {
        float left = final_intervals[i];
        float right = final_intervals[i+1];

        // Используем метод Ньютона. Начальное приближение - середина интервала
        // Нужно проверить смену знака, чтобы гарантировать наличие корня
        if (valP(left, p) * valP(right, p) <= 0) {
            float x0 = (left + right) / 2.0f;
            // Для надежности можно использовать метод секущих или бисекции сначала, 
            // но по заданию - Ньютон.
            // Проверим условие Фурье (f(x0)*f''(x0)>0) или просто запустим Ньютона
            float root = newtonMethod(x0, valP, valdP, p);
            
            // Проверка, что значение в корне действительно близко к 0
            if (fabs(valP(root, p)) < 1e-3) {
                 bool exists = false;
                 for(float r : p_roots) if(fabs(r - root) < 1e-4) exists = true;
                 if(!exists) {
                    p_roots.push_back(root);
                    cout << "Найден корень (собственное число) lambda = " << root << endl;
                 }
            }
        }
    }

    if (p_roots.empty()) {
        cout << "Вещественных собственных чисел не найдено (возможно, комплексно-сопряженные пары)." << endl;
        return 0;
    }

    // 3. Вычисление собственного вектора для первого найденного корня
    float lambda = p_roots[0];
    cout << "\n--- Вычисление собственного вектора для lambda = " << lambda << " ---" << endl;

    // Вектор y в базисе Фробениуса: (lambda^3, lambda^2, lambda, 1)^T
    vector<float> y(N);
    y[0] = pow(lambda, 3);
    y[1] = pow(lambda, 2);
    y[2] = pow(lambda, 1);
    y[3] = 1.0f;

    cout << "Вектор y (в базисе Фробениуса): ( ";
    for(float val : y) cout << val << " ";
    cout << ")^T" << endl;

    // Переход к вектору u в исходном базисе: u = S * y
    // S = M_{n-1} * ... * M_1.
    // В M_list у нас хранятся матрицы в порядке добавления: M3 (i=3), M2 (i=2), M1 (i=1).
    // По формуле x = M_3 * M_2 * M_1 * y.
    // Умножаем последовательно: u = M_3 * (M_2 * (M_1 * y))
    
    vector<float> u = y;
    // Идем с конца списка (M1) к началу (M3), так как умножение вектора справа: S*y = M3*M2*M1*y
    // Сначала M1*y, потом M2*res, потом M3*res.
    // В M_list[0] лежит M3, в M_list[2] лежит M1.
    // Значит цикл должен быть обратным по вектору M_list
    for (int k = M_list.size() - 1; k >= 0; k--) {
        u = multiplyMV(M_list[k], u);
    }

    cout << "Собственный вектор u: ( ";
    for(float val : u) cout << val << " ";
    cout << ")^T" << endl;

    // Проверка Au - lambda*u approx 0
    vector<float> Au = multiplyMV(A, u);
    vector<float> lu(N);
    for(int i=0; i<N; ++i) lu[i] = lambda * u[i];

    cout << "Проверка невязки (Au - lambda*u):" << endl;
    float max_err = 0;
    for(int i=0; i<N; ++i) {
        float err = fabs(Au[i] - lu[i]);
        if(err > max_err) max_err = err;
        cout << "  [" << i << "]: " << Au[i] << " - " << lu[i] << " = " << (Au[i] - lu[i]) << endl;
    }
    
    if(max_err < 1e-3) cout << "Результат: ВЕРНО (невязка мала)." << endl;
    else cout << "Результат: Большая погрешность (возможно, накопление ошибок float)." << endl;

    return 0;
}