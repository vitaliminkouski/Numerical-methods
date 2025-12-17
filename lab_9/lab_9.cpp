#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>

using namespace std;

// === Константы ===
const int N = 4;
const float EPS = 1e-5; // Точность для метода Ньютона
const float SEARCH_LIMIT = 500.0f; // Границы поиска корней [-LIMIT, +LIMIT]

typedef vector<vector<float>> Matrix;

// === Матричные операции (из ЛР 7) ===

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

vector<float> multiplyMV(const Matrix& A, const vector<float>& v) {
    int n = A.size();
    vector<float> res(n, 0.0);
    for(int i=0; i<n; ++i)
        for(int j=0; j<n; ++j)
            res[i] += A[i][j] * v[j];
    return res;
}

// Метод Данилевского
bool danilevskyMethod(Matrix A, Matrix& Phi, vector<Matrix>& M_list, vector<float>& p) {
    int n = A.size();
    Phi = A;
    M_list.clear();

    for (int i = n - 1; i > 0; i--) {
        float pivot = Phi[i][i - 1];
        if (fabs(pivot) < 1e-8) return false;

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

// === Функции Многочленов ===

// P(x) = x^4 - p1*x^3 - p2*x^2 - p3*x - p4
float valP(float x, const vector<float>& p) {
    return pow(x, 4) - p[0]*pow(x, 3) - p[1]*pow(x, 2) - p[2]*x - p[3];
}

// P'(x) = 4x^3 - 3p1*x^2 - 2p2*x - p3
float valdP(float x, const vector<float>& p) {
    return 4*pow(x, 3) - 3*p[0]*pow(x, 2) - 2*p[1]*x - p[2];
}

// P''(x) = 12x^2 - 6p1*x - 2p2
float valddP(float x, const vector<float>& p) {
    return 12*pow(x, 2) - 6*p[0]*x - 2*p[1];
}

// === Численные методы ===

// Метод Ньютона
// func - функция, корни которой ищем
// dfunc - её производная
float newtonMethod(float x0, float (*func)(float, const vector<float>&), float (*dfunc)(float, const vector<float>&), const vector<float>& p, bool& ok) {
    float x = x0;
    int iter = 0;
    ok = true;
    while (iter < 100) {
        float fVal = func(x, p);
        float dfVal = dfunc(x, p);
        if (fabs(dfVal) < 1e-12) { ok = false; return x; } // Производная 0, метод застрял
        
        float x_new = x - fVal / dfVal;
        if (fabs(x_new - x) < EPS) return x_new;
        
        x = x_new;
        iter++;
    }
    // Если не сошлось за 100 шагов
    if (fabs(func(x, p)) > 0.1) ok = false; 
    return x;
}

// Вспомогательная функция для проверки уникальности корня
bool isNewRoot(float val, const vector<float>& roots) {
    for (float r : roots) {
        if (fabs(r - val) < 1e-3) return false;
    }
    return true;
}

int main() {
    setlocale(LC_ALL, "Russian");
    srand(static_cast<unsigned int>(time(0)));

    cout << "========================================================" << endl;
    cout << "   ЛР №9: Метод Ньютона (Подробный режим вычислений)   " << endl;
    cout << "========================================================" << endl << endl;
    
    // --- 1. Генерация и Метод Данилевского ---
    Matrix A(N, vector<float>(N));
    Matrix Phi;
    vector<Matrix> M_list;
    vector<float> p; 
    bool success = false;

    do {
        for(int i=0; i<N; ++i)
            for(int j=0; j<N; ++j)
                A[i][j] = (rand()%21 - 10) + (float)rand()/RAND_MAX; 
        
        success = danilevskyMethod(A, Phi, M_list, p);
    } while (!success);

    printMatrix("1. Исходная матрица A", A);
    
    cout << "2. Результат метода Данилевского (Коэффициенты):" << endl;
    for(size_t i=0; i<p.size(); ++i) cout << "   p" << i+1 << " = " << p[i] << endl;
    cout << endl;

    // --- Вывод формул многочленов ---
    cout << "3. Анализ функций и производных:" << endl;
    cout << fixed << setprecision(2);
    cout << "   P(x)   = x^4 - (" << p[0] << ")*x^3 - (" << p[1] << ")*x^2 - (" << p[2] << ")*x - (" << p[3] << ")" << endl;
    cout << "   P'(x)  = 4x^3 - " << 3*p[0] << "*x^2 - " << 2*p[1] << "*x - " << p[2] << endl;
    cout << "   P''(x) = 12x^2 - " << 6*p[0] << "*x - " << 2*p[1] << endl << endl;

    // --- 4. Решение P''(x) = 0 (Степень 2) ---
    cout << "--------------------------------------------------------" << endl;
    cout << "ЭТАП A: Решение уравнения 2-й степени P''(x) = 0" << endl;
    cout << "   Уравнение: " << 12 << "x^2 + (" << -6*p[0] << ")x + (" << -2*p[1] << ") = 0" << endl;
    
    float a_sq = 12;
    float b_sq = -6 * p[0];
    float c_sq = -2 * p[1];
    float D = b_sq*b_sq - 4*a_sq*c_sq;
    vector<float> ddp_roots;

    if (D >= 0) {
        float r1 = (-b_sq - sqrt(D)) / (2*a_sq);
        float r2 = (-b_sq + sqrt(D)) / (2*a_sq);
        ddp_roots.push_back(r1);
        ddp_roots.push_back(r2);
        sort(ddp_roots.begin(), ddp_roots.end());
        cout << "   Дискриминант D = " << D << " >= 0. Найдены 2 корня." << endl;
        cout << "   >>> Корни P''(x): " << ddp_roots[0] << "; " << ddp_roots[1] << endl;
    } else {
        cout << "   Дискриминант D < 0. Корней нет. P'(x) монотонна." << endl;
    }
    cout << endl;

    // --- 5. Решение P'(x) = 0 (Степень 3) ---
    cout << "--------------------------------------------------------" << endl;
    cout << "ЭТАП B: Решение уравнения 3-й степени P'(x) = 0" << endl;
    cout << "   Интервалы поиска определяются корнями P''(x)." << endl;

    vector<float> search_intervals_dp = {-SEARCH_LIMIT};
    search_intervals_dp.insert(search_intervals_dp.end(), ddp_roots.begin(), ddp_roots.end());
    search_intervals_dp.push_back(SEARCH_LIMIT);

    vector<float> dp_roots;
    
    for (size_t i = 0; i < search_intervals_dp.size() - 1; ++i) {
        float left = search_intervals_dp[i];
        float right = search_intervals_dp[i+1];
        
        cout << "   Поиск на интервале [" << setw(7) << left << "; " << setw(7) << right << "] ... ";
        
        // Проверка смены знака (если знаки одинаковые, корня может не быть или их 2)
        // Для кубического уравнения и Ньютона можно пробовать середину
        float start_pt = (left + right) / 2.0f;
        if (fabs(start_pt) > SEARCH_LIMIT) start_pt = (left > -SEARCH_LIMIT) ? left + 1 : right - 1;

        bool ok;
        // Решаем P'(x)=0, используя производную P''(x)
        float root = newtonMethod(start_pt, valdP, valddP, p, ok);
        
        if (ok && root > left && root < right && isNewRoot(root, dp_roots)) {
            cout << "Найден корень: " << root << endl;
            dp_roots.push_back(root);
        } else {
            cout << "Корней нет или метод разошелся." << endl;
        }
    }
    sort(dp_roots.begin(), dp_roots.end());
    
    cout << "   >>> Корни P'(x) (Степень 3): ";
    if (dp_roots.empty()) cout << "нет вещественных корней";
    else for(float r : dp_roots) cout << r << "  ";
    cout << endl << endl;

    // --- 6. Решение P(x) = 0 (Степень 4) ---
    cout << "--------------------------------------------------------" << endl;
    cout << "ЭТАП C: Решение уравнения 4-й степени P(x) = 0 (Собственные числа)" << endl;
    cout << "   Интервалы поиска определяются корнями P'(x)." << endl;

    vector<float> search_intervals_p = {-SEARCH_LIMIT};
    search_intervals_p.insert(search_intervals_p.end(), dp_roots.begin(), dp_roots.end());
    search_intervals_p.push_back(SEARCH_LIMIT);

    vector<float> p_roots;

    for (size_t i = 0; i < search_intervals_p.size() - 1; ++i) {
        float left = search_intervals_p[i];
        float right = search_intervals_p[i+1];

        // Проверяем смену знака на концах интервала
        float valL = valP(left, p);
        float valR = valP(right, p);

        cout << "   Интервал [" << setw(7) << left << "; " << setw(7) << right << "]";
        
        // У полиномов на бесконечности значения огромные, ограничим вывод
        if (fabs(valL) > 1e6) valL = (valL > 0 ? 1.0f : -1.0f) * 999999;
        if (fabs(valR) > 1e6) valR = (valR > 0 ? 1.0f : -1.0f) * 999999;

        cout << " -> Знаки P(x): (" << (valP(left, p) > 0 ? "+" : "-") << " ... " << (valP(right, p) > 0 ? "+" : "-") << ") ";

        if (valP(left, p) * valP(right, p) <= 0) {
            bool ok;
            // Старт из середины
            float start_pt = (left + right) / 2.0f;
            // Решаем P(x)=0, используя производную P'(x)
            float root = newtonMethod(start_pt, valP, valdP, p, ok);
            
            if (ok && isNewRoot(root, p_roots)) {
                // Дополнительная проверка, что мы внутри интервала (или рядом)
                cout << "-> Найден корень: " << root << endl;
                p_roots.push_back(root);
            } else {
                 cout << "-> Сбой метода Ньютона." << endl;
            }
        } else {
            cout << "-> Смены знака нет (корней нет)." << endl;
        }
    }

    cout << endl;
    cout << "   >>> ИТОГОВЫЕ КОРНИ P(x) (Собственные числа): ";
    if (p_roots.empty()) cout << "Вещественных корней не найдено.";
    else for(float r : p_roots) cout << r << "  ";
    cout << endl << endl;

    // --- 7. Собственный вектор ---
    if (!p_roots.empty()) {
        float lambda = p_roots[0];
        cout << "--------------------------------------------------------" << endl;
        cout << "ЭТАП D: Проверка для первого корня lambda = " << lambda << endl;

        vector<float> y(N);
        y[0] = pow(lambda, 3);
        y[1] = pow(lambda, 2);
        y[2] = lambda;
        y[3] = 1.0f;

        vector<float> u = y;
        for (int k = M_list.size() - 1; k >= 0; k--) {
            u = multiplyMV(M_list[k], u);
        }

        cout << "   Восстановленный вектор u: ( ";
        for(float val : u) cout << val << " ";
        cout << ")^T" << endl;

        // Проверка
        vector<float> Au = multiplyMV(A, u);
        cout << "   Вектор невязки (Au - lambda*u):" << endl;
        for(int i=0; i<N; ++i) {
            float lu = lambda * u[i];
            cout << "     [" << i << "]: " << setw(10) << Au[i] << " - " << setw(10) << lu << " = " << scientific << (Au[i] - lu) << defaultfloat << endl;
        }
    }

    return 0;
}