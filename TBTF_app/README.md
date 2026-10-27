# Paradigma "Too Big to Fail" — Ghid teoretic

> Modelarea matematică a dependenței strategice de un actor dominant într-o rețea de aprovizionare GPU.
> Aplicație: Machine Learning pentru estimarea cererii + Problema Transporturilor pentru optimizarea costului + Algoritmul Ford-Fulkerson pentru capacitatea fizică a rețelei.

---

## Cuprins

1. [Motivație și problema generală](#1-motivație-și-problema-generală)
2. [Estimarea cererii prin Random Forest](#2-estimarea-cererii-prin-random-forest)
3. [Problema Transporturilor (PT)](#3-problema-transporturilor-pt)
   - 3.1 [Formularea generală](#31-formularea-generală)
   - 3.2 [Echilibrarea problemei](#32-echilibrarea-problemei)
   - 3.3 [Soluție inițială: metoda Colțului Nord-Vest](#33-soluție-inițială-metoda-colțului-nord-vest)
   - 3.4 [Optimizare: algoritmul MODI](#34-optimizare-algoritmul-modi)
   - 3.5 [Tehnica perturbării ε](#35-tehnica-perturbării-ε)
4. [Flux în rețele și Ford-Fulkerson](#4-flux-în-rețele-și-ford-fulkerson)
   - 4.1 [Graf rețea și flux](#41-graf-rețea-și-flux)
   - 4.2 [Tăietură și teorema Ford-Fulkerson](#42-tăietură-și-teorema-ford-fulkerson)
   - 4.3 [Algoritmul de marcaj](#43-algoritmul-de-marcaj)
5. [Combinarea celor două modele](#5-combinarea-celor-două-modele)
6. [Studiu de caz: ecosistemul GPU](#6-studiu-de-caz-ecosistemul-gpu)
   - 6.1 [Scenariul 1 — Piața normală](#61-scenariul-1--piața-normală)
   - 6.2 [Scenariul 2 — Colaps NVIDIA](#62-scenariul-2--colaps-nvidia)
7. [Paradoxul costului aparent](#7-paradoxul-costului-aparent)
8. [Implementare](#8-implementare)
9. [Extensii și pipeline scalabil](#9-extensii-și-pipeline-scalabil)

---

## 1. Motivație și problema generală

Conceptul de **„Too Big to Fail" (TBTF)** descrie situația în care un actor economic devine atât de important pentru un sistem încât eșecul său ar produce daune sistemice. Origine: criza financiară 2007-2008, în care anumite bănci au fost considerate prea importante pentru a fi lăsate să cadă.

În ecosistemul tehnologic actual, dependența infrastructurii AI globale de NVIDIA prezintă caracteristici similare. Întrebarea pe care o adresăm este:

> **Cum cuantificăm matematic vulnerabilitatea unei rețele de aprovizionare atunci când un actor dominant devine indisponibil parțial sau total?**

Pentru a răspunde, descompunem problema în trei pași:

1. **Estimarea cererii viitoare** pe regiuni — prin Random Forest pe serii istorice
2. **Optimizarea alocării cost-minime** — prin Problema Transporturilor (PTE)
3. **Verificarea fezabilității fizice** — prin algoritmul Ford-Fulkerson

Fiecare model răspunde la o întrebare diferită, iar combinarea lor expune fenomenul TBTF într-un mod cuantificabil.

---

## 2. Estimarea cererii prin Random Forest

### Motivație

Vectorul cererii **B = (b₁, b₂, ..., bₙ)** care alimentează modelul de optimizare nu este postulat arbitrar. Este derivat din date istorice prin regresie.

### Modelul

Pentru fiecare regiune r ∈ {SUA, Germania, Japonia, China, România}, datele istorice 2018-2024 sunt modelate ca:

$$D_r(t) = \beta_r \cdot e^{\alpha_r (t - 2018)} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

unde:
- $\beta_r$ = volumul inițial de bază
- $\alpha_r$ = rata de creștere anuală (între 15,5% și 17,5%, aleasă astfel încât seria să ajungă la cererea din studiul de caz)
- $\epsilon$ = zgomot gaussian care reflectă fluctuațiile pieței

### Random Forest Regressor

Un model **Random Forest** cu următorii hiperparametri:
- 200 de arbori (`n_estimators=200`)
- adâncime maximă 4 (`max_depth=4`)
- seed fixat pentru reproductibilitate (`random_state=42`)

este antrenat, pentru fiecare regiune, pe creșterea logaritmică de la un an la altul din seria 2018-2024 (creșterea anului t → creșterea anului t+1) și este aplicat de două ori, pentru 2025 și 2026. Valoarea pentru 2026 se rotunjește la multipli de 50 u.p.

> Seriile istorice sunt **sintetice** (zgomot gaussian de 2%): modelul ilustrează mecanismul de estimare, nu piața reală.

### Justificare metodologică

| Avantaj RF | Detaliu |
|------------|---------|
| Robustețe la outlieri | Mediana predicțiilor de arbori reduce impactul valorilor extreme |
| Captează non-linearități | Fără presupunerea unui model specific (liniar, exponențial) |
| Estimarea incertitudinii | Variabilitatea între arbori dă banda de încredere |
| Anti-overfitting | `max_depth=4` pe 7 puncte istorice — suficient de simplu |

### Alternative considerate

- **ARIMA**: presupune staționaritate, problematic pe serii cu creștere exponențială
- **LSTM**: over-engineering pentru o serie de 7 puncte (peste 100 parametri vs 7 observații)
- **Regresie polinomială**: tendință de overfitting la grad ≥ 3

### Output

Pentru 2026, modelul produce vectorul cererii **B = (300, 150, 200, 250, 100)** cu Σ bⱼ = 1000 u.p., care devine input al pasului următor.

---

## 3. Problema Transporturilor (PT)

### 3.1 Formularea generală

Considerăm:
- **m** centre de producție (surse) **Aᵢ**, i = 1..m, cu disponibilitățile **aᵢ ≥ 0**
- **n** destinații (piețe) **Bⱼ**, j = 1..n, cu necesarele **bⱼ ≥ 0**
- Costul unitar de transport **cᵢⱼ ≥ 0** de la Aᵢ la Bⱼ

Variabila de decizie xᵢⱼ reprezintă cantitatea transportată de la sursa i la destinația j.

**Forma generală (FG):**

$$\min \, f(x) = \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij} \cdot x_{ij}$$

subiectă la:

$$\sum_{j=1}^{n} x_{ij} \leq a_i, \quad i = \overline{1, m} \quad \text{(disponibilitate)}$$

$$\sum_{i=1}^{m} x_{ij} \geq b_j, \quad j = \overline{1, n} \quad \text{(cerere)}$$

$$x_{ij} \geq 0, \quad \forall i, j \quad \text{(non-negativitate)}$$

### 3.2 Echilibrarea problemei

Problema este **echilibrată (PTE)** dacă:

$$\sum_{i=1}^{m} a_i = \sum_{j=1}^{n} b_j$$

În acest caz, toate restricțiile devin egalități și soluția optimă este garantată să existe (Teorema fundamentală a PT).

Dacă problema este **neechilibrată**, se transformă în PTE prin introducerea unui actor fictiv:

- **Deficit de ofertă** (Σa < Σb): se introduce **furnizor fictiv** Aₘ₊₁ cu disponibil aₘ₊₁ = Σbⱼ − Σaᵢ și costuri cₘ₊₁,ⱼ = 0
- **Excedent de ofertă** (Σa > Σb): se introduce **destinație fictivă** Bₙ₊₁ cu necesar bₙ₊₁ = Σaᵢ − Σbⱼ și costuri cᵢ,ₙ₊₁ = 0

> **Atenție conceptuală:** Furnizorul/destinația fictiv(ă) este un artefact matematic. Unitățile alocate acolo nu există fizic — sunt doar variabile care satisfac restricția de echilibru cu cost zero. **Acest fapt este central pentru paradoxul costului aparent.**

**Forma standard (PTES):**

$$\min \, f(x) = \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij} \cdot x_{ij}$$

subiectă la:

$$\sum_{j=1}^{n} x_{ij} = a_i, \quad i = \overline{1, m}$$

$$\sum_{i=1}^{m} x_{ij} = b_j, \quad j = \overline{1, n}$$

$$x_{ij} \geq 0, \quad \sum_{i} a_i = \sum_{j} b_j$$

### Teoreme fundamentale

| Teoremă | Enunț |
|---------|-------|
| **T1** | Orice PTE are cel puțin o soluție posibilă |
| **T2** | Orice PTE admite cel puțin o soluție optimă |
| **T3** | Rangul matricei sistemului PTES este `m + n − 1` |

O **soluție de bază** este una cu cel mult m + n − 1 variabile nenule. Dacă numărul componentelor nenule este exact m + n − 1, soluția este **nedegenerată**; altfel este **degenerată**.

### 3.3 Soluție inițială: metoda Colțului Nord-Vest

Pentru a porni algoritmul de optimizare, avem nevoie de o soluție inițială fezabilă. **Metoda Colțului Nord-Vest** este cea mai simplă metodă:

```
1. Începe în celula (1, 1) (colțul nord-vest)
2. Alocă x₁₁ = min(a₁, b₁)
3. Scade x₁₁ din disponibilul aᵢ și necesarul bⱼ
4. Dacă aᵢ epuizat → coboară pe rândul următor (i+1, j)
   Dacă bⱼ epuizat → muta pe coloana următoare (i, j+1)
5. Repetă până când toate aᵢ și bⱼ sunt zero
```

**Avantaj:** Simplitate algoritmică, garantat să producă o soluție de bază.
**Dezavantaj:** Nu ia în considerare costurile cᵢⱼ — soluția inițială este de obicei departe de optim. De aceea avem nevoie de pasul următor.

### 3.4 Optimizare: algoritmul MODI

**MODI** (Modified Distribution Method) verifică optimalitatea unei soluții și produce direcția de îmbunătățire dacă aceasta nu este optimă.

#### Problema duală

Pentru PTES, problema duală (PTED) este:

$$\max \, g(u, v) = \sum_{i=1}^{m} a_i u_i + \sum_{j=1}^{n} b_j v_j$$

subiectă la:

$$u_i + v_j \leq c_{ij}, \quad \forall i, j$$

$$u_i, v_j \in \mathbb{R}$$

#### Teorema ecartelor complementare

Între soluția optimă a PTES și PTED există relația:

$$x_{ij} \cdot (c_{ij} - u_i - v_j) = 0, \quad \forall i, j$$

Aceasta se traduce în: pentru orice variabilă bazică (xᵢⱼ > 0) avem **uᵢ + vⱼ = cᵢⱼ**.

#### Algoritmul

**Pas 1.** Pornește cu soluția inițială X obținută prin Colțul Nord-Vest. Identifică mulțimea celulelor bazice J = {(i,j) | xᵢⱼ > 0}.

**Pas 2.** Calculează **potențialele** prin rezolvarea sistemului:

$$u_i + v_j = c_{ij}, \quad (i, j) \in J$$

Sistemul are m + n necunoscute și m + n − 1 ecuații, deci o necunoscută rămâne liberă. Convenția uzuală: **u₁ = 0**, restul se calculează prin propagare.

**Pas 3.** Calculează **costurile modificate** pentru celulele nebazice:

$$\tilde{c}_{ij} = u_i + v_j, \quad (i, j) \notin J$$

**Pas 4.** Calculează **diferențele**:

$$\delta_{ij} = c_{ij} - \tilde{c}_{ij} = c_{ij} - (u_i + v_j)$$

**Pas 5. Criteriu de optimalitate:**

- Dacă **toate δᵢⱼ ≥ 0** → soluția X este **optimă**. STOP.
- Dacă **∃ (i,j) ∉ J cu δᵢⱼ < 0** → soluția nu este optimă, se continuă cu pasul 6.

**Pas 6.** Se selectează celula de pivotare:

$$\delta_{p,q} = \min_{(i,j) \notin J} \{\delta_{ij} < 0\}$$

**Pas 7.** Se construiește **ciclul** care pleacă din (p,q), trece doar prin celule bazice și revine în (p,q). Acest ciclu este unic.

**Pas 8.** Se marchează celulele ciclului cu semne alternative: (+θ), (−θ), (+θ), (−θ)...

**Pas 9.** Se calculează:

$$\theta = \min \{x_{ij} \mid (i,j) \text{ celulă de rang impar a ciclului}\}$$

**Pas 10.** Se operează transportul θ de-a lungul ciclului:
- Celulele cu (+θ): xᵢⱼ + θ
- Celulele cu (−θ): xᵢⱼ − θ

**Pas 11.** Se obține soluția nouă X' cu cost îmbunătățit:

$$f'_{min} = f_{min} + \delta_{p,q} \cdot \theta$$

(notă: δ\_{p,q} este negativ, deci costul scade)

**Pas 12.** Se reia procedura de la Pasul 2.

#### Convergență

Algoritmul converge în număr finit de iterații deoarece:
- Există un număr finit de soluții de bază posibile (combinări de m+n−1 celule din mn celule)
- Fiecare iterație produce o soluție cu cost strict mai mic (pentru probleme nedegenerate)
- Deci nu se poate face o buclă infinită

### 3.5 Tehnica perturbării ε

**Problema soluțiilor degenerate:** Dacă numărul de componente nenule NC < m + n − 1, apare **fenomenul de ciclaj** — algoritmul poate reveni la o soluție anterioară fără să facă progres.

**Soluție:** Tehnica perturbării adaugă un mic ε > 0 la valorile aᵢ și bₙ:

$$a_i' = a_i + \epsilon, \quad i = \overline{1, m}, \qquad b_n' = b_n + m\epsilon, \qquad 0 \leq \epsilon \ll 1$$

Această perturbare garantează că orice combinație Σaᵢ în submulțimi I₁ ⊂ I și Σbⱼ în J₁ ⊂ J este distinctă, eliminând degeneracia.

Se aplică algoritmul MODI pe PTE_ε, iar în soluția finală se face **ε = 0**.

---

## 4. Flux în rețele și Ford-Fulkerson

Problema Transporturilor presupune că rețeaua de transport poate transmite efectiv orice cantitate cerută. **Această ipoteză poate fi falsă în practică** — există constrângeri fizice de capacitate. Pentru a verifica fezabilitatea, folosim modelul fluxului în rețele.

### 4.1 Graf rețea și flux

**Definiția 1 (graf finit conex).** Graful G = (X, U) este finit conex dacă X este o mulțime finită și pentru orice x, y ∈ X, x ≠ y, există un lanț (succesiune de arce adiacente) care le unește.

**Definiția 2 (graf rețea).** Graful G = (X, U) finit conex și fără bucle se numește **graf rețea** dacă există:
- un vârf **xₛ ∈ X** numit **sursă**, cu Γ⁻¹(xₛ) = ∅ (în xₛ nu ajunge niciun arc)
- un vârf **xₜ ∈ X** numit **destinație**, cu Γ(xₜ) = ∅ (din xₜ nu pleacă niciun arc)

**Definiția 3 (capacitate).** Pentru U mulțimea arcelor grafului, definim funcția:

$$c: U \rightarrow \mathbb{R}_+$$

unde c(u) este **capacitatea arcului** u.

**Definiția 4 (flux).** Funcția φ: U → ℝ₊ se numește **flux** în graful rețea G dacă satisface trei condiții:

**(i) Capacitate:**

$$0 \leq \varphi(u) \leq c(u), \quad \forall u \in U$$

**(ii) Conservare** (la noduri intermediare):

$$\sum_{x_i \in \Gamma_{x_k}} \varphi(x_k, x_i) = \sum_{x_j \in \Gamma^{-1}_{x_k}} \varphi(x_j, x_k), \quad \forall k \neq s, t$$

(suma fluxurilor care ies dintr-un nod = suma fluxurilor care intră)

**(iii) Valoarea fluxului:**

$$\sum_{x_i \in \Gamma_{x_s}} \varphi(x_s, x_i) = \sum_{x_j \in \Gamma^{-1}_{x_t}} \varphi(x_j, x_t) = v(\varphi)$$

**Definiția 5 (arc saturat).** Un arc u ∈ U se numește:
- **saturat** dacă φ(u) = c(u)
- **nesaturat** dacă φ(u) < c(u)

### 4.2 Tăietură și teorema Ford-Fulkerson

**Definiția 6 (tăietură).** Submulțimea A ⊂ X, A ≠ ∅ se numește **tăietură** în graful rețea G dacă xₛ ∉ A și xₜ ∈ A.

Tăietura partiționează vârfurile în două submulțimi disjuncte: S = X \ A (conține sursa) și T = A (conține destinația).

**Definiția 7 (mulțimea arcelor tăieturii).**

$$U_A^- = \{u = (x_i, x_j) \mid x_i \notin A, \, x_j \in A\}$$

**Definiția 8 (capacitatea tăieturii).**

$$c(U_A^-) = \sum_{u \in U_A^-} c(u)$$

### Teorema Ford-Fulkerson

> **Într-un graf rețea G, valoarea maximă a fluxului coincide cu capacitatea minimă a tăieturilor sale:**
>
> $$\max_{\varphi \in \mathcal{F}} v(\varphi) = \min_{A \subset \mathcal{T}} c(U_A^-)$$

**Demonstrație (schiță).** Pentru orice flux φ și orice tăietură A:
- Tot fluxul de la xₛ la xₜ trebuie să străbată tăietura A
- Valorile fluxului pe arce nu pot depăși capacitățile arcelor
- Deci **v(φ) ≤ c(U_A⁻)** pentru orice combinație (φ, A)

Egalitatea se atinge atunci când tăietura A este saturată de fluxul φ (toate arcele tăieturii au φ(u) = c(u)). ∎

**Consecință practică:** Pentru a găsi fluxul maxim, este suficient să identificăm o tăietură de capacitate minimă. Algoritmul Ford-Fulkerson face acest lucru constructiv.

### 4.3 Algoritmul de marcaj

**Pas 1. Flux inițial.** Se determină un flux inițial φ₀ ∈ F alegând câteva drumuri de la xₛ la xₜ și saturându-le. Cel mai simplu: φ₀(u) = 0 pentru toate arcele.

**Pas 2. Procedura de marcaj** (testul de optimalitate):

- Se marchează sursa **xₛ cu [+]**
- Pentru fiecare arc:
  - Dacă (xᵢ marcat) → (xⱼ nesaturat marcat), se marchează xⱼ cu **[+xᵢ]**
  - Dacă (xᵢ nesaturat) ← (xⱼ marcat), se marchează xᵢ cu **[−xⱼ]**
  - Dacă (xᵢ) → (xⱼ saturat), arcul nu se folosește pentru marcare
- Se propagă marcajul prin graf

**Pas 3. Decizie:**

- **Dacă destinația xₜ nu poate fi marcată** → fluxul φ este **optim**, algoritmul STOP. Tăietura minimă este mulțimea nodurilor marcate.
- **Dacă xₜ a fost marcată** → fluxul nu este optim, continuă cu pasul 4.

**Pas 4. Reconstituire lanț.** Folosind marcajele, se reconstituie lanțul μ de la xₛ la xₜ (urmărind marcajele invers).

**Pas 5. Cantitatea de îmbunătățire.** De-a lungul lanțului μ, se calculează:

$$\alpha = \min \{c(u) - \varphi(u) \mid u \in \mu\}$$

**Pas 6. Update flux.** Se obține fluxul îmbunătățit:

$$\varphi'(u) = \begin{cases} \varphi(u) + \alpha & \text{dacă } u \in \mu \\ \varphi(u) & \text{altfel} \end{cases}$$

cu valoarea **v(φ') = v(φ) + α**.

**Pas 7.** Se reia pasul 2 cu noul flux φ'.

#### Convergență

Algoritmul converge deoarece:
- Fiecare iterație crește v(φ) cu α > 0
- v(φ) este mărginit de min c(U_A⁻) (tăietura minimă)
- Pentru capacități întregi, α este întreg, deci numărul iterațiilor este finit

#### Observație importantă

Ordinea în care se aleg drumurile **nu influențează rezultatul final** v(φ\*), doar valorile intermediare în cadrul unei iterații. Acest fapt garantează că algoritmul produce optimul global indiferent de strategia de selecție a lanțurilor.

---

## 5. Combinarea celor două modele

PTE și Ford-Fulkerson răspund la **întrebări diferite** despre aceeași rețea:

| Aspect | Problema Transporturilor (PTE) | Ford-Fulkerson (FF) |
|--------|--------------------------------|---------------------|
| **Întrebare** | „Cum aloc cel mai ieftin?" | „Cât pot transporta efectiv?" |
| **Output** | X* (alocare optimă), f* (cost minim) | v(φ*) (flux maxim), tăietură T |
| **Presupune** | Rețeaua poate transmite toată cererea | Doar capacitățile fizice ale arcelor |
| **Limitare** | Acceptă soluții fictive în neechilibru | Nu poate fi „înșelat" — flux real |

**Combinarea lor produce o optimizare completă** a fluxului de distribuție:
- ✓ Cost optim
- ✓ Validare fezabilitate fizică
- ✓ Cuantificare deficit (când există)
- ✓ Identificare bottleneck (prin tăietura minimă)

---

## 6. Studiu de caz: ecosistemul GPU

### Setup

- **Furnizori:** NVIDIA, AMD, Intel Arc (m = 3)
- **Piețe:** SUA, Germania, Japonia, China, România (n = 5)
- **Matrice cost** (u.m./u.p.):

$$
C = \begin{pmatrix}
2 & 5 & 4 & 6 & 7 \\
3 & 4 & 6 & 5 & 6 \\
5 & 6 & 7 & 4 & 5
\end{pmatrix}
$$

- **Vector cerere** (din ML pentru 2026): **B = (300, 150, 200, 250, 100)**, Σ bⱼ = 1000 u.p.

### 6.1 Scenariul 1 — Piața normală

**Capacități:** NVIDIA = 500, AMD = 300, Intel Arc = 200 → Σ aᵢ = 1000

**Echilibrul:** Σ aᵢ = Σ bⱼ = 1000 → problema este echilibrată direct, fără furnizor fictiv.

**Soluția inițială** (Colțul Nord-Vest): cost **4 100 u.m.**

**Soluția optimă** (o singură iterație MODI: intră AMD → Germania cu δ = −3, θ = 150, iese AMD → Japonia):

| X* | SUA | Germania | Japonia | China | România | Σ aᵢ |
|----|-----|----------|---------|-------|---------|------|
| **NVIDIA** | 300 | 0 | 200 | 0 | 0 | 500 |
| **AMD** | 0 | 150 | 0 | 150 | 0 | 300 |
| **Intel Arc** | 0 | 0 | 0 | 100 | 100 | 200 |
| **Σ bⱼ** | 300 | 150 | 200 | 250 | 100 | 1000 |

Celulele NVIDIA → China, NVIDIA → România și AMD → România au δ = 0, deci există **optime alternative**: alte planuri cu același cost.

**Rezultate:**

| Indicator | Valoare |
|-----------|---------|
| Cost optim PTE (f\*) | **3 650 u.m.** (verificat cu HiGHS) |
| Flux maxim Ford-Fulkerson (v(φ)) | **1 000 u.p.** |
| Acoperire cerere | **100% ✓** |
| Deficit fizic | 0 |
| Capacitate tăietură c(T) | 1 000 = v(φ) ✓ |

### 6.2 Scenariul 2 — Colaps NVIDIA

**Capacități:** NVIDIA = 100 (reducere drastică), AMD = 300, Intel Arc = 200 → Σ aᵢ real = 600

**Echilibrul:** Σ aᵢ = 600 < Σ bⱼ = 1000 → problema este **neechilibrată**, necesită **furnizor fictiv A\* cu 400 u.p. și cost 0**.

**Soluția inițială** (Colțul Nord-Vest): cost **2 550 u.m.**

**Soluția optimă** (două iterații MODI: intră Intel Arc → China cu δ = −3, θ = 150 → 2 100 u.m.; apoi A\* → Germania cu δ = −2, θ = 50 → 2 000 u.m.):

| X* | SUA | Germania | Japonia | China | România | Σ aᵢ |
|----|-----|----------|---------|-------|---------|------|
| **NVIDIA** | 100 | 0 | 0 | 0 | 0 | 100 |
| **AMD** | 200 | 100 | 0 | 0 | 0 | 300 |
| **Intel Arc** | 0 | 0 | 0 | 200 | 0 | 200 |
| **A\* (fictiv)** | 0 | 50 | 200 | 50 | 100 | 400 |
| **Σ bⱼ** | 300 | 150 | 200 | 250 | 100 | 1000 |

Deficitul se concentrează în Japonia (100%) și România (100%); Germania pierde 33%, China 20%, iar SUA este servită integral.

**Rezultate:**

| Indicator | Valoare |
|-----------|---------|
| Cost optim PTE (f\*) | **2 000 u.m.** (aparent mai mic cu 45%!) |
| Flux maxim Ford-Fulkerson (v(φ)) | **600 u.p.** (adevărul fizic) |
| Acoperire cerere | **60%** |
| Deficit fizic | **400 u.p.** (40% piață neaprovizionată) |
| Capacitate tăietură c(T) | 600 = v(φ) ✓ |
| Bottleneck | Toate cele 3 arce sursă → furnizori sunt saturate |

---

## 7. Paradoxul costului aparent

### Faptul matematic

Cost optim PTE:
- S1: f\* = **3 650 u.m.**
- S2: f\* = **2 000 u.m.** (paradoxal **mai mic**, cu 45%)

### Capcana interpretativă

Dacă judecăm doar pe baza funcției obiectiv f\*, criza pare să **îmbunătățească** situația. PTE este matematic corect — minimizează exact ceea ce trebuie să minimizeze. Concluzia naivă bazată doar pe cost este însă **greșită**.

### De ce apare paradoxul

Cele **400 u.p.** care lipsesc din producția fizică sunt absorbite de furnizorul fictiv cu cost 0. PTE nu „știe" că aceste unități **nu există** — pentru algoritm sunt doar variabile care satisfac restricția de echilibru cu cost minim. Costul aparent scade tocmai pentru că **40% din alocare este gratuită și fictivă**.

### Rolul Ford-Fulkerson

Ford-Fulkerson nu permite această iluzie. Operează exclusiv pe rețeaua fizică, fără furnizori fictivi. Răspunsul său **v(φ) = 600 u.p.** este adevărul observabil: oferta totală reală e 600, ceea ce înseamnă deficit de 400 u.p. — **exact volumul mascat de PTE**.

### Identitatea cheie

$$\boxed{\underbrace{v(\varphi^\ast)}_{\text{flux real}} + \underbrace{\sum_j x^\ast_{m+1,j}}_{\text{volum fictiv}} = \underbrace{\sum_j b_j}_{\text{cererea totală}}}$$

Pentru S2: **600 + 400 = 1000** ✓

Sau echivalent:

$$\boxed{\text{Deficit fizic} = \sum_j b_j - v(\varphi^\ast) = \text{Volum furnizor fictiv}}$$

### Legătura cu paradigma TBTF

Concentrarea capacității la un singur actor dominant (NVIDIA = 50% din ofertă în S1) creează o vulnerabilitate care **nu se vede în costurile nominale** ale rețelei. Modelele financiare tradiționale subestimează riscul sistemic exact din acest motiv.

### Testul de stres

Pentru fiecare furnizor se simulează 20 000 de căderi aleatoare (seed 2026): partea pierdută din capacitate θ ~ U(0,2; 1), rezervele celorlalți σᵢ ~ U(0; 0,2) și variația cererii εⱼ ~ U(−0,1; 0,1). Deficitul cauzat de cădere se măsoară față de același scenariu fără cădere.

| Furnizor | Importanță sistemică SI | Mediana | Percentila 95 | Absorbit de ceilalți | P(pierdere > 20%) |
|----------|------------------------|---------|---------------|----------------------|-------------------|
| NVIDIA | **25,0%** | 25,1% | 43,5% | 31,8% | 62,7% |
| AMD | 11,2% | 11,0% | 23,6% | 51,6% | 14,2% |
| Intel Arc | 5,0% | 4,1% | 14,0% | 68,8% | 0,1% |

NVIDIA are de 1,67 ori capacitatea AMD, dar căderea ei costă de 2,23 ori mai mult; față de Intel Arc rapoartele sunt 2,5 și 4,99. Codul este în `stress_test.py` (`python3 stress_test.py` reproduce tabelul).

> **Concluzia teoretică:** Vulnerabilitatea TBTF este **cuantificabilă matematic** ca distanța dintre fluxul fizic maxim și cererea totală, sau echivalent, ca volumul furnizorului fictiv introdus pentru echilibrare.

---

## 8. Implementare

### Stack tehnologic

- **Python 3.10+**
- **Streamlit** — interfață web interactivă
- **NumPy / Pandas** — calcul matricial și manipulare date
- **SciPy** (`linprog`, `milp`, metoda HiGHS) — verificarea costului optim și agentul de recuperare
- **scikit-learn** (`RandomForestRegressor`) — predicție cerere
- **Plotly** — grafice interactive (linii, bare, heatmap, Sankey)
- **Graphviz (DOT, randat direct de Streamlit)** — rețeaua de flux și schemele logice

### Structura aplicației

```
ML-AFF.py            interfața Streamlit (în limba engleză)
├── Sidebar: set de date (instanța din articol 500/300/200, cote reale 880/120/0,
│            date personalizate), scenariul S1/S2, cursorul de capacitate NVIDIA
├── Overview: diagramă Sankey, acoperirea cererii pe piețe, paradoxul costului
├── Demand (ML): prognoza Random Forest pe regiuni
├── Transport (cost): Colțul Nord-Vest, iterațiile MODI, potențiale, verificare HiGHS
├── Maximum flow: rețeaua cu tăietura minimă, drumurile Edmonds–Karp
├── Stress test: Algoritmul 1 (Monte Carlo) și analiza de sensibilitate
├── Recovery agent: prototip multi-perioadă cu loturi și piețe prioritare
├── Who depends on NVIDIA: expunerea marilor cumpărători (Omdia, 2024)
└── Method and theory: scheme logice și ghid teoretic
tbtf_core.py         modelele (fără interfață): RF, Colțul N-V, MODI, Edmonds–Karp, agent
stress_test.py       testul de stres din articol
.streamlit/config.toml  tema aplicației
```

### Modul de utilizare

```bash
# Instalare dependențe
pip install -r requirements.txt

# Rulare
streamlit run ML-AFF.py
```

### Pseudocod algoritmi cheie

**Random Forest pentru cerere:**
```python
for fiecare regiune r:
    g = diff(log(cerere_istorica[r]))          # creșterea logaritmică anuală
    model = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(g[:-1], g[1:])                   # creșterea anului t -> creșterea anului t+1
    B[r] = 50 * round(cerere_2026[r] / 50)     # după două aplicări (2025, 2026)
```

**PTE (Colțul Nord-Vest + MODI, verificat cu HiGHS):**
```python
C, A, B = echilibreaza(C, A, B)                # furnizor sau piață fictivă, cost 0
X, baza = colt_nord_vest(A, B)                 # m+n-1 celule de bază
while True:
    u, v = potentiale(C, baza)                 # uᵢ + vⱼ = cᵢⱼ pe celulele de bază
    delta = C - (u + v.T)                      # pe celulele nebazice
    if all(delta >= 0): break                  # plan optim
    (p, q) = argmin(delta)                     # celula care intră
    ciclu = ciclu_prin_baza(baza, (p, q))
    theta = min(X pe pozițiile (−) ale ciclului)
    X = X ± theta pe ciclu; actualizează baza
assert cost(X) == linprog(..., method="highs").fun
```

**Ford-Fulkerson (BFS pentru augmenting paths):**
```python
while bfs_path(rezidual_graph, source, sink) exists:
    alpha = min(rezidual_graph[u][v] pentru (u,v) în path)
    actualizează graf rezidual cu ±alpha
    max_flow += alpha
return max_flow, min_cut_S, min_cut_T
```

---

## 9. Extensii și pipeline scalabil

Arhitectura demo-ului prezent (3 furnizori × 5 piețe × 1 produs) este versiunea minimă a unei platforme decizionale extensibile.

### Arhitectura propusă pentru scalare

```
[ DATE SURSĂ ]  →  [ INGEST ]  →  [ ML LAYER ]  →  [ OPTIM LAYER ]  →  [ DECISION ]
   N companii        ETL /         RF per           PTE + FF              Dashboard
   M piețe           streaming     produs            (paralel)             agregat
   K produse                       × regiune                               + alerts
```

### Componente

- **Date sursă:** baze multiple cu serii temporale, capacități, prețuri, contracte
- **Ingest:** ETL (Airflow/dbt), normalizare, agregare, validare
- **ML Layer:** Random Forest per (produs, regiune), produce vectori cerere B
- **Optim Layer:** PTE + FF în paralel, câte o instanță per produs
- **Decision:** Dashboard agregat cu alerts pe pragul critic (capacități sub care apare deficit)

### Aplicații în afara GPU

Structura matematică este **generică**. Aceeași arhitectură poate fi aplicată pentru:

| Domeniu | Actor dominant | Produs |
|---------|----------------|--------|
| Semiconductori | TSMC | Chip-uri 3nm/5nm |
| Energie | Gazprom (pre-2022) | Gaz natural |
| Cloud | AWS | Capacitate compute |
| Farma | Pfizer | Vaccinuri specifice |
| Cobalt | RDC (mining) | Materii prime baterii |

În toate cazurile, dependența unui actor dominant produce vulnerabilitate sistemică cuantificabilă prin modelul nostru.

---

## Copyright

© 2026 Universitatea Națională de Știință și Tehnologie POLITEHNICA București (UNSTPB)
Facultatea de Științe Aplicate

Toate drepturile rezervate. Materialul este proprietatea autorilor și a instituției coordonatoare. Reproducerea sau distribuirea integrală sau parțială fără acordul scris al autorilor este interzisă.

## Autori

- Dedu Anișoara-Nicoleta
- Dumitrescu Andreea-Mihaela
- Iliescu Daria
- Lungu Diana-Ionela

**Universitatea Națională de Știință și Tehnologie POLITEHNICA București**
Facultatea de Științe Aplicate · Anul III · Grupa 1333a
Coordonator: **Lect. Dr. Simona Mihaela BIBIC**

---

> *„Paradigma «Too Big to Fail» nu este doar o intuiție economică — este un fenomen matematic, cuantificabil prin distanța v(φ\*) − Σbⱼ."*
