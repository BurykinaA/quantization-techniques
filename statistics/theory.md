## 1. MinMaxObserver

Для набора активаций $\{x_i\}$ находим:

$$
x_\text{min} = \min_i x_i, 
\quad
x_\text{max} = \max_i x_i.
$$

Пусть $Q_\text{min}, Q_\text{max}$ — границы целевой квантизованной шкалы (например, для 8-бит unsigned: $0\ldots255$). Тогда параметры квантизации:

$$
\text{scale} 
= \frac{x_\text{max} - x_\text{min}}{Q_\text{max} - Q_\text{min}}, 
\quad
\text{zero\_point}
= \operatorname{round}\Bigl(Q_\text{min} - \frac{x_\text{min}}{\text{scale}}\Bigr).
$$

При квантовании значение $x$ переводится в:

$$
q = \operatorname{clamp}\bigl(\operatorname{round}(x / \text{scale}) + \text{zero\_point},\,Q_\text{min},Q_\text{max}\bigr).
$$

---

## 2. PerChannelMinMaxObserver

То же самое, но параметры считаются **отдельно для каждого канала** $c$:

$$
x_{\min}^{(c)} = \min_{i\,:\,\text{channel}(i)=c} x_i,
\quad
x_{\max}^{(c)} = \max_{i\,:\,\text{channel}(i)=c} x_i.
$$

Затем для каждого канала:

$$
\text{scale}^{(c)} = \frac{x_{\max}^{(c)} - x_{\min}^{(c)}}{Q_\text{max} - Q_\text{min}},
\quad
\text{zero\_point}^{(c)} 
= \operatorname{round}\Bigl(Q_\text{min} - \frac{x_{\min}^{(c)}}{\text{scale}^{(c)}}\Bigr).
$$

Квантуем по той же формуле $q^{(c)}$.

---

## 3. HistogramObserver (с KL-divergence или MSE-критерием)

### 1. Сбор гистограммы

Накопим $H$ — гистограмму значений $x_i$ на $B$ равных бинов в диапазоне $[a,b]$.

### 2. Поиск оптимального порога

Предположим, что мы хотим ограничить границы квантизации $[L, U] \subset [a, b]$. Тогда:

- усекаем все $x < L$ в $L$, а $x > U$ в $U$,
- строим «сглаженную» квантованную гистограмму $H_q$ при $k$ уровнях,
- выбираем $(L,U)$ так, чтобы минимизировать меру расхождения $\mathcal{D}(H\;\|\;H_q)$.

#### a) KL-divergence

$$
(L^*,U^*) = \arg\min_{L<U}
\; D_{\mathrm{KL}}\bigl(P\;\|\;Q\bigr)
= \arg\min_{L<U}
\sum_i P(i)\,\log\frac{P(i)}{Q(i)},
$$

где $P$ — нормированная гистограмма оригинала, $Q$ — гистограмма после клиппинга+квантизации.

#### b) MSE-criterion

Альтернативно можно минимизировать MSE между оригиналом и квантизированными значениями:

$$
(L^*,U^*) = \arg\min_{L<U}
\frac{1}{N}\sum_{i=1}^N \bigl(x_i - Q_{\!LU}(x_i)\bigr)^2.
$$

После выбора $(L^*,U^*)$, параметры scale/zero_point считаются как в MinMaxObserver, но с $x_\text{min}=L^*$, $x_\text{max}=U^*$.

---

## 4. MSEObserver (перебор порогов)

Этот Observer напрямую перебирает кандидаты $(L,U)$ (например, беря пороги по процентилям гистограммы) и для каждого вычисляет:

$$
\text{MSE}(L,U)
= \frac{1}{N}\sum_i\Bigl(x_i - \operatorname{dequantize}\bigl(\operatorname{quantize}_{L,U}(x_i)\bigr)\Bigr)^2.
$$

И выбирает:

$$
(L^*,U^*) = \arg\min_{L<U} \text{MSE}(L,U).
$$

---

### Когда что использовать

- **MinMaxObserver**  
  — быстро и дешёво, если выбросов мало; 8-бит на «хорошо себя ведущих» активациях.

- **PerChannelMinMaxObserver**  
  — всегда для весов; для активаций, когда каналы сильно различаются по динамике.

- **HistogramObserver (KL или MSE-mode)**  
  — при наличии длинных хвостов или при низкой битности (4–6 бит), чтобы отсечь редкие экстремумы без сильного искажения основной массы.

- **MSEObserver**  
  — только когда нужны самые тонкие настройки, и вы готовы тратить время на перебор порогов.

Эти формулы помогут вам понимать, какие статистики собирают разные Observer’ы и как они влияют на итоговую точность квантизации.
