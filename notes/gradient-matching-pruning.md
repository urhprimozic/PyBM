# Gradient matching: interval zaupanja za hitro rezanje kandidatnih enačb

*Kako iz ENE, poceni izvedbe gradient matchinga (malo naključno vzorčenih testnih točk, GP
interpolant namesto resničnega $x$) dobiti pošten interval zaupanja za pravo napako kandidatne
strukture, in ga uporabiti za rezanje očitno slabih kandidatk med strukturnim iskanjem.
Spremljevalni dokument k `gradient-matching.md` - ista notacija ($\hat x$, $e(t)$, $\ell$, GP
definicije), tu ponovljena samo na kratko, s sklicem nazaj za polno izpeljavo.*

---

## 0. Notacija (nadaljevanje `gradient-matching.md`)

Za kandidatno enačbo $x'=F(x,t,\theta)$, GP interpolant $\hat x$ (fiksen, glej `gradient-matching.md`
Definicija 1.2) in testno točko $t_i$:

- $r(t_i,\theta) := \hat x'(t_i) - F(\hat x(t_i),t_i,\theta)$ - gradient-matching residual (Definicija
  2.1 v `gradient-matching.md`),
- $\rho_i(\theta) := g(r(t_i,\theta))$ - *penalty* ($g(r)=|r|$ za L1, $g(r)=r^2$ za L2),
- $L_n(\theta) := \frac1n\sum_{i=1}^n \rho_i(\theta)$ - gradient-matching izguba na $n$ naključno
  vzorčenih testnih točkah (**ne** nujno enaka mreži $t_1,\dots,t_m$ iz `gradient-matching.md` §2 -
  tu $n$ namenoma majhen, izbran za hiter presejalni prehod),
- $\hat\theta_n := \arg\min_\theta L_n(\theta)$,
- $\sigma^2(t):=\mathrm{Var}[X(t)\mid\text{podatki}]$, $\sigma_d^2(t):=\mathrm{Var}[X'(t)\mid\text{podatki}]$ -
  GP posteriorni varianci stanja in odvoda (glej `gradient-matching.md` Definicija 1.2'),
- $F_x(t,\theta) := \partial_x F(\hat x(t),t,\theta)$ - Jacobijan desne strani po stanju,
- $\ell$ - korelacijska dolžina GP-jevega jedra.

Cilj: iz $L_n(\hat\theta_n)$ (poceni, na malo točkah) oceniti interval, v katerem se z visoko
verjetnostjo nahaja "polna" napaka te strukture - tista, ki bi jo dobili z vsemi razpoložljivimi
$N\gg n$ točkami IN z resničnim $x$ namesto $\hat x$.

---

## 1. Dva neodvisna vira negotovosti

Ključno je ločiti dve POPOLNOMA RAZLIČNI vprašanji.

**Vir A - vzorčenje.** Katerih $n$ od razpoložljivih $N$ točk smo izbrali? Vprašanje o
*deterministični* količini $\rho_i(\theta)$ pri fiksnem $\hat x$ - naključnost izhaja samo iz tega,
kateri $t_i$ so pristali v vzorcu.

**Vir B - interpolacija.** Je resnični $X(t_i)$ sploh enak $\hat x(t_i)$? Ne vprašanje o vzorčenju -
$t_i$ je fiksen, naključnost je v GP-jevem POSTERIORJU o tem, kje je resnično stanje.

Pomembno: ne gre za "ponovi cel eksperiment z novim šumom" (frekventistična varianca cenilke čez
hipotetične ponovitve podatkov) - $\hat x$ ostaja skozi vse spodaj fiksen, izračunan enkrat iz
ENEGA, že opazovanega nabora podatkov. Oba vira spodaj sta o tem istem, fiksnem naboru.

---

## 2. Vzorčna komponenta (Vir A)

**Definicija 2.1 (vzorčna varianca).**
$$s^2(\theta) = \frac{1}{n-1}\sum_{i=1}^n\big(\rho_i(\theta)-\bar\rho(\theta)\big)^2$$
(Besselov popravek $n-1$, nepristranska cenilka.)

**Trditev 2.2 (CLT za $L_n$).** Za dovolj velik $n$ (in ne prehudo poševne $\rho_i$) je $L_n(\theta)$
približno normalno porazdeljen okoli svoje "polne" ($N$-točkovne) vrednosti:
$$\mathrm{Var}_A[L_n(\theta)] \approx \frac{s^2(\theta)}{n}\Big(1-\frac nN\Big)$$
(popravek za končno populacijo $(1-n/N)$ zanemarljiv, kadar $n\ll N$). To je natanko Centralni
limitni izrek: povprečje $n$ (grobo neodvisnih) spremenljivk je približno normalno NE GLEDE na
obliko posameznih $\rho_i$.

**Opomba (neodvisnost ni samoumevna).** Če so testne točke blizu skupaj, GP-jeva korelacija med
$\hat x(t_i),\hat x(t_j)$ pada kot $e^{-\Delta t^2/2\ell^2}$ in ni zanemarljiva. Rešitev: testne
točke naključno vzorčimo ČEZ interval dolžine $\gg\ell$ - tedaj je razmik med naključnima točkama
tipično $\Delta t\gg\ell$, neodvisnost pa velja praktično zastonj.

---

## 3. Interpolacijska komponenta (Vir B)

GP posterior natanko pove, kako negotov je $(X(t_i),X'(t_i))$ glede na podatke:
$$\begin{pmatrix}X(t_i)\\X'(t_i)\end{pmatrix}\Big|\text{podatki}\ \sim\ \mathcal N\left(\begin{pmatrix}\hat x(t_i)\\\hat x'(t_i)\end{pmatrix},\ \begin{pmatrix}\sigma^2(t_i) & c(t_i)\\ c(t_i) & \sigma_d^2(t_i)\end{pmatrix}\right)$$

$\sigma^2,\sigma_d^2$ poznamo (`GP1dim.std`, `GP1dim.std_derivative`, glej `odestimate`). Manjka
prečni člen $c(t_i):=\mathrm{Cov}[X(t_i),X'(t_i)]$ - kovarianca med stanjem in odvodom PRI ISTI
točki.

**Trditev 3.1 (prečni člen).** Za stacionarno jedro je apriorna kovarianca med $f(t)$ in $f'(t)$
pri isti $t$ natanko 0 (odvod sode funkcije pri 0 je 0). Posteriorna ni nujno 0 - oba se
pogojujeta na iste podatke:
$$c(t) = -v(t)^\top v'(t)$$
kjer sta $v(t)=L^{-1}k(t,X_{\text{train}})^\top$, $v'(t)=L^{-1}\partial_tk(t,X_{\text{train}})^\top$
natanko projekciji, ki ju `std`/`std_derivative` že računata. En dodaten skalarni produkt - nič
novega ni treba zgraditi.

*Dokaz (skica).* Splošna formula posteriorne kovariance: $\mathrm{Cov}_{\text{post}}[f(t),g(t')] =
\mathrm{Cov}_{\text{prior}}[f(t),g(t')] - v_f(t)^\top v_g(t')$. Za $f=\mathrm{id}$, $g=$ odvod,
$t'=t$: prvi člen izgine (stacionarnost + sodost jedra), ostane drugi. $\blacksquare$

**Delta metoda - propagacija skozi $F$.** Linearizacija (ista kot povsod v `gradient-matching.md`
§1.4 - velja natanko toliko, kolikor je $\sigma(t_i)$ res majhna, tj. testne točke so izbrane v
"dobrem" delu domene):
$$r_{\text{true}}(t_i)-r(t_i) \approx \delta X'(t_i) - F_x(t_i,\theta)\cdot\delta X(t_i)$$
Linearna kombinacija skupno-Gaussovih $(\delta X,\delta X')$ je spet Gaussova, z varianco (skalarno
stanje):
$$\tau^2(t_i) := \sigma_d^2(t_i) - 2F_x(t_i,\theta)\,c(t_i) + F_x(t_i,\theta)^2\,\sigma^2(t_i)$$

Za penalty $\rho=g(r)$, z verižnim pravilom $\delta\rho\approx g'(r)\cdot\delta r$:

| penalty | $g(r)$ | $\mathrm{Var}[\delta\rho_i]$ |
|---|---|---|
| L1 | $\lvert r\rvert$ | $\tau^2(t_i)$ |
| L2 | $r^2$ | $4\,r(t_i)^2\,\tau^2(t_i)$ |

**Trditev 3.2 (interpolacijska varianca izgube).** Ker so testne točke (Vir A) izbrane narazen
($\Delta t\gg\ell$), so tudi $\delta\rho_i$ med sabo neodvisne - vsota neodvisnih Gaussovih je
EKSAKTNO Gaussova (brez dodatnega CLT-približka na tem koraku):
$$\mathrm{Var}_B[L_n(\theta)] \approx \frac{1}{n^2}\sum_{i=1}^n g'(r_i)^2\,\tau^2(t_i)$$

---

## 4. Skupna negotovost in interval zaupanja

Vir A in Vir B sta pojmovno ločena (eno je "katere točke", drugo "je $\hat x$ enak resničnemu $x$
pri fiksni točki"), in oba prispevata s pričakovano vrednostjo nič prvega reda (linearizacija
okoli posteriorne SREDINE, po definiciji brez pristranskosti) - zato se preprosto seštejeta.

**Trditev 4.1 (skupna varianca in interval zaupanja).**
$$\mathrm{Var}[L_n(\hat\theta_n)] \approx \mathrm{Var}_A + \mathrm{Var}_B = \frac{s^2}{n} + \frac{1}{n^2}\sum_{i=1}^n g'(r_i)^2\tau^2(t_i)$$
$$\mathrm{SE} = \sqrt{\mathrm{Var}[L_n(\hat\theta_n)]}$$
$$\text{interval zaupanja} = L_n(\hat\theta_n) \ \pm\ t_{n-1}(1-\alpha/2)\cdot\mathrm{SE}$$

Studentova $t_{n-1}$ namesto normalne $z$, ker je $n$ namenoma majhen - konsistentno konservativno
(širše, ne ožje, kot pri velikem $n$).

---

## 5. Algoritem (za eno kandidatno strukturo)

1. Fitaj GP $\hat x(t)$ na podatke (enkrat, deli se med vsemi kandidatkami).
2. Izberi interval z nizko, enakomerno $\sigma(t)$ (izven vrzeli v podatkih), dolžine $\gg\ell$.
3. Naključno vzorči $n$ testnih točk znotraj tega intervala.
4. Fitaj hitro: $\hat\theta_n = \arg\min_\theta L_n(\theta)$.
5. Iz že-obstoječih rezidualov $\rho_i$ izračunaj $\bar\rho\,(=L_n)$ in $s^2$ (Vir A).
6. Za vsak $t_i$ izračunaj $F_x(t_i,\hat\theta_n)$, $\sigma(t_i)$, $\sigma_d(t_i)$, $c(t_i)$
   $\Rightarrow \tau^2(t_i)$ (Vir B).
7. Sestavi $\mathrm{SE}=\sqrt{s^2/n + \frac1{n^2}\sum g'(r_i)^2\tau^2(t_i)}$.
8. Interval: $L_n(\hat\theta_n)\pm t_{n-1}(1-\alpha/2)\cdot\mathrm{SE}$.

**Rezanje kandidatk.** Strukturo A odstrani v prid strukturi B samo, če se intervala NE prekrivata:
$$L_n^A - t^*\mathrm{SE}^A \;>\; L_n^B + t^*\mathrm{SE}^B \quad\Longrightarrow\quad \text{odstrani A}$$
Kandidatke, ki eksplodirajo za VSAK $\theta$ (loss ogromen na peščici razpršenih $\theta$, brez
optimizacije), odstrani že PRED tem korakom - brezplačen, ločen prvi filter.

---

## 6. Omejitve

- Vse zgoraj je delta metoda (linearizacija prvega reda) - velja natanko toliko, kolikor je
  $\sigma(t_i)$ res majhna na izbranem intervalu. Drugi red ($F_{xx}\cdot\sigma^2$) daje majhno
  dodatno pristranskost, ki tu ni zajeta - empirično (Bled) je ta zunaj vrzeli v podatkih ~1 %
  linearnega člena, znotraj vrzeli pa lahko preseže 100 % (glej `gradient-matching.md` §1.4 za isti
  fenomen).
- Kombinirana pivotalna količina (Vir A ocenjen iz vzorca, Vir B znan analitično) ni strogo
  Studentova $t$ - uporaba $t_{n-1}$ na SKUPNEM SE je praktičen, rahlo konservativen približek, ne
  eksakten rezultat.
- Namen je hitro in VARNO rezanje očitno slabih kandidatk, ne natančna statistika - raje širši
  (konservativen) SE kot preozek. Napačno zavržena dobra kandidatka je draga napaka; nekaj krogov
  dlje obdržana slaba kandidatka je poceni.
