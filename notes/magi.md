# MAGI: gradient matching z neopazovanimi spremenljivkami

*Kako deluje MAGI (MAnifold-constrained Gaussian process Inference, Yang, Wong, Kou 2021), kako ga
prevedemo iz Bayesovskega vzorčenja v optimizacijo (MAP, "maximum likelihood") in kako bi ga
implementirali v `odestimate`. Spremljevalni dokument k `gradient-matching.md`: tam je GP
interpolant fiksen in izračunan samo iz podatkov, tu postanejo vrednosti trajektorije
neznanke, kar omogoči tudi spremenljivke brez podatkov.*

---

## Kako brati ta dokument

- Formule, označene z **(MAGI, JSS §…)**, so prepisane iz članka o paketu MAGI
  (Wong, Yang, Kou, *J. Stat. Softw.* 109(4), 2024; arXiv 2203.06066), ki metodo povzame z istimi
  oznakami kot izvirni članek v PNAS.
- **Trditve z dokazom** so naše izpeljave. Kjer je trditev preverjena tudi numerično (na kodi v
  `odestimate`), je to navedeno.
- Kjer gre za hevristiko ali odprto vprašanje, je to izrecno označeno.

**Povzetek v treh stavkih.**
1. MAGI da GP prior *vsaki* komponenti stanja, tudi neopazovani, in vrednosti trajektorije na mreži
   $I$ obravnava kot neznanke, skupaj s $\theta$.
2. ODE vstopi kot pogoj, da je GP-jev odvod v točkah $I$ enak $f$. Iz pravil pogojne verjetnosti
   sledi zaprta oblika posteriorja, brez numerične integracije in brez proste uteži "$\lambda$".
3. Če posterior namesto vzorčenja maksimiziramo, dobimo nelinearen problem najmanjših kvadratov
   velikosti $D|I|+p$, ki ga rešuje Gauss-Newton. Ima pa tri degeneracije, ki jih moramo poznati
   (§5.4).

---

## 0. Notacija

| simbol | pomen |
|---|---|
| $x(t)\in\mathbb R^D$ | stanje sistema; $x_d(t)$ je komponenta $d$ |
| $\dot x(t)$ | časovni odvod |
| $f(x,\theta,t)\in\mathbb R^D$ | desna stran ODE, $\dot x = f(x,\theta,t)$ (v `gradient-matching.md`: $F$, parametri $c$) |
| $\theta\in\Omega_\theta\subset\mathbb R^p$ | parametri (konstante modela) |
| $\tau_d$ | časi meritev komponente $d$ (prazna množica, če je $d$ neopazovana) |
| $y_d(\tau_d)$ | meritve, $y_d(t) = x_d(t)+\varepsilon$, $\varepsilon\sim\mathcal N(0,\sigma_d^2)$ |
| $I=\{t_1,\dots,t_n\}$ | diskretizacijska mreža, $\tau_d\subset I$ za vse $d$ |
| $x_d(I)\in\mathbb R^{|I|}$ | vektor vrednosti $(x_d(t_1),\dots,x_d(t_n))$ |
| $f_d(x(I),\theta,t_I)\in\mathbb R^{|I|}$ | vektor $\big(f(x(t_i),\theta,t_i)_d\big)_{i=1}^n$ |
| $\mathcal K_d(s,t)$ | kovariančna funkcija (jedro) GP-ja za komponento $d$ |
| $\mu_d(t)$ | povprečna funkcija priorja (privzeto 0) |
| $\phi_d$ | hiperparametri jedra (varianca, korelacijska dolžina) |
| $\|v\|_A^2$ | $v^\top A v$ |
| $D_o, D_\ell$ | indeksi opazovanih in neopazovanih (latentnih) komponent |

Odvodi jedra (MAGI, JSS §4.2):
$$
{}'\mathcal K(s,t) = \frac{\partial}{\partial s}\mathcal K(s,t),\qquad
\mathcal K'(s,t) = \frac{\partial}{\partial t}\mathcal K(s,t),\qquad
\mathcal K''(s,t) = \frac{\partial^2}{\partial s\,\partial t}\mathcal K(s,t).
$$
Za stacionarno jedro $\mathcal K(s,t)=\kappa(s-t)$ velja ${}'\mathcal K(s,t)=\kappa'(s-t)$,
$\mathcal K'(s,t)=-\kappa'(s-t)$ in $\mathcal K''(s,t)=-\kappa''(s-t)$. V `odestimate` je
${}'\mathcal K$ = `RBFWithNoise.gradient_X`, $\mathcal K''$ = `RBFWithNoise.hessian_XY`.

---

## 1. Zakaj navaden gradient matching tu ne zadošča

Gradient matching iz `gradient-matching.md` naredi dva ločena koraka:

1. Iz podatkov komponente $d$ izračuna GP posteriorno povprečje $\hat x_d(t)$ in njegov odvod
   $\hat x_d'(t)$. Ti dve funkciji sta od tu naprej **fiksni**.
2. Poišče $\theta$, ki minimizira $\sum_i\|\hat x'(t_i) - f(\hat x(t_i),\theta,t_i)\|^2$.

Za neopazovano komponento korak 1 ni mogoč: brez podatkov ni $\hat x_\ell$ in ne $\hat x_\ell'$.
Ker $f$ v splošnem bere vse komponente (npr. plenilec v enačbi plena), ostanek tudi za opazovane
komponente ni izračunljiv.

Avtorji MAGI izpostavijo še bolj načelen problem prejšnjih GP pristopov (Calderhead et al. 2009,
Dondelinger et al. 2013, Wenk et al. 2019): $\dot x$ je tam podan **dvakrat in nezdružljivo**,
enkrat z GP-jem (kot odvod interpolanta) in enkrat z ODE ($f$). Ta dva opisa so nato združevali s
hevristikami (npr. produktom dveh gostot z dodatnim parametrom tolerance). MAGI namesto tega
GP *pogoji* na to, da njegov odvod zadošča ODE, kar je ena sama koherentna verjetnostna konstrukcija
(MAGI, JSS §2).

---

## 2. Gradniki: GP in njegov odvod

### 2.1 Skupna porazdelitev vrednosti in odvoda

**Definicija 2.1.** Komponenti $d$ damo GP prior $X_d\sim\mathcal{GP}(\mu_d,\mathcal K_d)$,
komponente med sabo neodvisne. Če je $\mathcal K_d$ dovolj gladko jedro (§3.7), je proces
v srednjem kvadratnem odvedljiv, $\dot X_d$ je spet GP, in ker je odvajanje linearna operacija, je par
$(X_d(I),\dot X_d(I))$ skupno Gaussovski:
$$
\begin{pmatrix} X_d(I)\\ \dot X_d(I)\end{pmatrix}
\sim\mathcal N\!\left(
\begin{pmatrix}\mu_d(I)\\ \dot\mu_d(I)\end{pmatrix},
\begin{pmatrix} \mathcal K_d(I,I) & \mathcal K'_d(I,I)\\ {}'\mathcal K_d(I,I) & \mathcal K''_d(I,I)\end{pmatrix}
\right). \tag{2.1}
$$
Kovarianci izhajata iz $\operatorname{Cov}[\dot X(s),X(t)] = \partial_s\mathcal K(s,t)$ in
$\operatorname{Cov}[\dot X(s),\dot X(t)]=\partial_s\partial_t\mathcal K(s,t)$ (Solak et al. 2003;
Rasmussen in Williams 2006, §9.4; enako kot (1.1) v `gradient-matching.md`). Ker je $\mathcal K$
simetrično, velja $\mathcal K'_d(I,I) = {}'\mathcal K_d(I,I)^\top$.

### 2.2 Odvod pri danih vrednostih

**Trditev 2.2** (MAGI, JSS §4.2). Pri danem $X_d(I)=x_d(I)$ velja
$$
\dot X_d(I)\mid X_d(I)=x_d(I)\;\sim\;\mathcal N\big(\dot\mu_d(I) + m_d\,\{x_d(I)-\mu_d(I)\},\ \Psi_d\big),
\tag{2.2}
$$
$$
C_d = \mathcal K_d(I,I),\qquad
m_d = {}'\mathcal K_d(I,I)\,C_d^{-1},\qquad
\Psi_d = \mathcal K''_d(I,I) - {}'\mathcal K_d(I,I)\,C_d^{-1}\,\mathcal K'_d(I,I).
$$
$C_d$ je kovarianca $x_d(I)$, $m_d$ preslika vrednosti v pričakovani odvod, $\Psi_d$ je kovarianca
odvoda, ki ostane, ko vrednosti poznamo.

*Dokaz.* Standardna formula za pogojno porazdelitev pri skupno Gaussovskem paru $(a,b)$:
$b\mid a\sim\mathcal N\big(\mu_b+\Sigma_{ba}\Sigma_{aa}^{-1}(a-\mu_a),\ \Sigma_{bb}-\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}\big)$
(Schurov komplement). Vstavimo $a=X_d(I)$, $b=\dot X_d(I)$ in bloke iz (2.1). $\blacksquare$

### 2.3 Kaj $m_d$ in $\Psi_d$ pomenita

**Trditev 2.3.** $\dot\mu_d(I)+m_d\{x_d(I)-\mu_d(I)\}$ je natanko odvod (v točkah $I$)
brezšumnega GP interpolanta skozi točke $(I, x_d(I))$.

*Dokaz.* GP, pogojen na $X_d(I)=x_d(I)$ brez šuma, ima povprečje
$\mu_d(t)+\mathcal K_d(t,I)\,C_d^{-1}\{x_d(I)-\mu_d(I)\}$. Odvajamo po $t$ in vstavimo $t\in I$. $\blacksquare$

$m_d$ je torej "odvajanje prek interpolacije", linearna operacija na vrednostih. To je natanko
tvoja ideja s "seedi": vrednosti v $I$ so seedi, $m_d$ da odvod interpolanta skozi njih.

**Trditev 2.4.** Če mrežo $I$ zgoščujemo okoli $t_i$, gre $(\Psi_d)_{ii}\to 0$.

*Dokaz.* Pogojna varianca je najmanjša srednja kvadratna napaka med vsemi ocenami iz $X_d(I)$, torej
tudi manjša od napake ocene z diferenčnim kvocientom: če $t_i\pm h\in I$, je
$(\Psi_d)_{ii}\le \mathbb E\big[\big(\dot X_d(t_i)-\tfrac{X_d(t_i+h)-X_d(t_i-h)}{2h}\big)^2\big]\to 0$
za $h\to 0$, ker je $\dot X_d$ odvod v srednjem kvadratnem. $\blacksquare$

Posledica za §3: teža ODE pogoja $\Psi_d^{-1}$ z gostoto mreže raste, gostejša mreža torej strožje
uveljavlja ODE. Hkrati $C_d$ in $\Psi_d$ postajata slabo pogojeni (§6.7).

---

## 3. MAGI

### 3.1 Mnogoterost rešitev ODE

(MAGI, JSS §2.) Množica trajektorij, ki zadoščajo ODE za neki $\theta$:
$$
\mathcal X=\big\{x:[0,T]\to\mathbb R^D\ \big|\ x_d\in C^1[0,T],\ \dot x(t)=f(x(t),\theta,t)\ \forall t\in[0,T],\ \text{za neki }\theta\in\Omega_\theta\big\}.
$$
Slučajna spremenljivka, ki meri odstopanje GP-jeve trajektorije od te množice:
$$
W=\sup_{t\in[0,T],\,d}\big|\dot X_d(t)-f(X(t),\theta,t)_d\big|.
$$
$W=0$ natanko tedaj, ko je realizacija $X$ rešitev ODE. Idealni posterior bi bil
$p(\theta,x\mid W=0,\,y(\tau))$.

### 3.2 Diskretizacija

Pogoja $W=0$ na zveznem intervalu ne znamo izračunati, zato ga nadomestimo z diskretnim:
$$
W_I=\max_{t\in I,\,d}\big|\dot X_d(t)-f(X(t),\theta,t)_d\big|,\qquad \tau\subset I .
$$
$W_I=0$ pomeni: odvod GP-ja je enak $f$ **v vsaki točki mreže**, med točkami pa ne nujno.

### 3.3 Posterior

(MAGI, JSS §2, enačba (3).) Po Bayesovem pravilu:
$$
p(\theta,x(I)\mid W_I=0,y(\tau))
\;\propto\; \pi(\theta)\;\times\; p(x(I))\;\times\; p(y(\tau)\mid x(I))\;\times\; p\big(\dot X(I)=f(x(I),\theta,t_I)\ \big|\ x(I)\big). \tag{3.1}
$$
Štirje faktorji:

1. $\pi(\theta)$ je prior parametrov.
2. $p(x(I))$ je GP prior na mreži, $\mathcal N(\mu_d(I), C_d)$ za vsako komponento. **Ne** vključuje
   podatkov.
3. $p(y(\tau)\mid x(I)) = p(y(\tau)\mid x(\tau))$ je verjetje meritev. Ker $\tau\subset I$, so
   $x(\tau)$ kar nekatere izmed neznank.
4. Gostota (2.2) odvoda pri danih vrednostih, **ovrednotena pri $\dot X(I)=f$**. Iz tega faktorja
   pride ODE.

Poenostavitve, ki jih članek uporabi: GP prior in meritve so neodvisni od $\theta$, odvod pri danih
vrednostih pa je pogojno neodvisen od $\theta$ in od meritev. Zato
$p(W_I=0\mid y,x(I),\theta) = p(\dot X(I)=f(x(I),\theta,t_I)\mid x(I))$.

**Opomba 3.1 (pogojevanje na dogodek z verjetnostjo 0; naša opomba).** $W_I=0$ ima verjetnost 0,
zato je "pogojevanje" nanj treba razumeti kot limito, rezultat pa je lahko odvisen od izbire limite
(Borel-Kolmogorov paradoks). Tu je enoličen. Naj bo $U=\dot X(I)-f(X(I),\theta,t_I)\in\mathbb R^{D|I|}$.
Velja $W_I=\|U\|_\infty$. Pri danih $(\theta,x(I))$ je $U$ le premik $\dot X(I)$, z Jacobijevo
determinanto 1, zato je pogojna gostota $U$ v $0$ enaka gostoti (2.2) v $f$. Če $W_I=0$ razumemo
kot limito dogodkov $\{\|U\|_\infty<\epsilon\}$, je njihova prostornina $(2\epsilon)^{D|I|}$ neodvisna
od $(\theta,x)$ in se v (3.1) okrajša. Isto bi dobili z $\|U\|_2<\epsilon$. Drugačen rezultat bi dobili
le, če bi pogoj zapisali z nelinearno transformacijo $U$, ki je odvisna od $(\theta, x)$.

### 3.4 Logaritem posteriorja

Pri fiksnih $\phi$ in $\sigma$ ter s temperiranjem $\beta$ iz §3.5:
$$
\boxed{
\begin{aligned}
\mathcal L(\theta,x(I)) := -\log p(\theta,x(I)\mid W_I=0,y)
= \sum_{d=1}^{D}\Big[
&\ \frac{1}{2\sigma_d^2}\big\|x_d(\tau_d)-y_d(\tau_d)\big\|^2
+\frac{1}{2\beta}\big\|x_d(I)-\mu_d(I)\big\|^2_{C_d^{-1}}\\
&+\frac{1}{2\beta}\big\|f_d(x(I),\theta,t_I)-\dot\mu_d(I)-m_d\{x_d(I)-\mu_d(I)\}\big\|^2_{\Psi_d^{-1}}
\Big]-\log\pi(\theta)+\text{konst.}
\end{aligned}}
\tag{3.2}
$$
$$
\text{konst.}=\sum_d\Big[\tfrac{|\tau_d|}{2}\log(2\pi\sigma_d^2)+\tfrac{1}{2\beta}\big(2|I|\log 2\pi+\log\det C_d+\log\det\Psi_d\big)\Big].
$$
Tri člene beremo takole:

- **podatki:** $x$ naj bo blizu meritev;
- **prior:** $x$ naj bo gladka trajektorija prave velikosti, kot jo dopušča jedro;
- **ODE:** odvod interpolanta skozi $x(I)$ naj bo enak $f$.

Vsi trije so kvadratne forme. Nelinearnost je samo v $f$.

Pomembno je, da $f_d$ bere vrednosti **vseh** komponent v istem času, zato ODE člen sklaplja
komponente. Prek tega sklopa pride informacija do neopazovanih komponent (§3.6).

### 3.5 Temperiranje $\beta$

(MAGI, JSS §4.1.) Člena 2 in 3 v (3.2) imata skupaj $2D|I|$ prispevkov, podatkovni člen pa le
$N=\sum_d|\tau_d|$. Z gostitvijo mreže bi GP člena prevladala nad podatki. MAGI zato produkt
$p(x(I))\,p(\dot X(I)=f\mid x(I))$ potencira na $1/\beta$, s priporočeno vrednostjo
$$
\beta = \frac{D\,|I|}{\sum_{d=1}^{D}|\tau_d|}
\qquad\text{(število točk mreže / število meritev, čez vse komponente)}.
$$
Če se $|I|$ podvoji, se prispevek GP členov na log-skali razpolovi.

- To je hevristika za stabilnost ocen pri zaporedju gostejših mrež, ne posledica verjetnostnega
  modela. Temperirani (3.1) ni več pravi posterior.
- **Neskladje v članku:** v seznamu nastavitev (`priorTemperature`, JSS §3) je privzeta vrednost
  opisana obratno, kot "število meritev / število točk mreže". Po §4.1 in po namenu (deli log-prior,
  zmanjša GP z gostejšo mrežo) je pravilno $D|I|/N$.
- Neopazovane komponente imajo $|\tau_d|=0$, zato povečajo $\beta$.

### 3.6 Neopazovane komponente

Za neopazovano komponento $\ell$ v (3.2) podatkovnega člena ni. Njen $x_\ell(I)$ je določen samo s
tremi stvarmi:

1. z njeno lastno ODE ($\dot x_\ell\approx f_\ell$);
2. s sklopom: $x_\ell$ nastopa v $f_d$ opazovanih komponent, katerih ODE člen je vezan na podatke;
3. s priorjem (gladkost, velikost).

Ali to zadošča za določitev $x_\ell$ in $\theta$, je vprašanje identifikabilnosti modela, ne metode.
Kjer model ni identifikabilen, se MAP obnaša slabo, glej §5.4.

### 3.7 Hiperparametri in jedro

(MAGI, JSS §2 in §4.2.) MAGI teče v treh korakih:

1. **Opazovane komponente:** $\phi_d$ in $\sigma_d$ dobi z GP fitom na podatke (MAP s šibkim
   normalnim priorjem na korelacijsko dolžino). Začetni $x_d(I)$ je linearna interpolacija meritev.
2. **Neopazovane komponente:** $\phi_\ell$, $x_\ell(I)$ in $\theta$ dobi s skupno optimizacijo (3.2),
   pri fiksnih vrednostih za opazovane komponente iz koraka 1. Brez neopazovanih komponent
   optimizira le $\theta$. To so začetne vrednosti za korak 3.
3. **HMC** vzorči $(\theta,x(I))$, in $\sigma$, če ni znan. $\phi$ je med vzorčenjem fiksen.

**Jedro.** Privzeto je Matérnovo jedro z $\nu=2.01$:
$$
\mathcal K(s,t)=\phi_1\,\frac{2^{1-\nu}}{\Gamma(\nu)}\Big(\sqrt{2\nu}\,\tfrac{r}{\phi_2}\Big)^{\nu}B_\nu\Big(\sqrt{2\nu}\,\tfrac{r}{\phi_2}\Big),\qquad r=|s-t|,
$$
kjer je $\phi_1$ varianca, $\phi_2$ korelacijska dolžina, $B_\nu$ pa modificirana Besslova funkcija
druge vrste.

Zakaj Matérn in ne RBF: za (2.1) mora $\dot X$ obstajati. Matérnov proces je $k$-krat
odvedljiv v srednjem kvadratnem natanko tedaj, ko $\nu>k$ (Rasmussen in Williams 2006, §4.2.1), za
en odvod torej zadošča $\nu>1$. MAGI izbere $\nu=2.01$ in navaja, da so gladkejša jedra (Matérn 5/2,
RBF) preveč gladka za ostre spremembe in vodijo v pristranske ocene $\theta$. Naš `RBFWithNoise` je
neskončno gladek, za watershed s `max(0, ·)`, if-else in dnevnimi skoki vhodov torej ravno najslabša
izbira (glej §6.6).

Še dve privzeti nastavitvi: povprečje priorja je 0, opcija `positiveSystem` pa vsili $x\ge 0$.

### 3.8 Vzorčenje (HMC)

Hamiltonov Monte Carlo vzorči iz gostote $\propto e^{-\mathcal L(z)}$, $z=(\theta,x(I))$. Na gibalni
količini $q\sim\mathcal N(0,M)$ simulira Hamiltonovo dinamiko z energijo $\mathcal L(z)+\tfrac12q^\top M^{-1}q$
v $L$ "leapfrog" korakih, vsak korak pa rabi gradient $\nabla\mathcal L$. Na koncu predlog sprejme ali
zavrne z Metropolisovim pravilom.

Privzete nastavitve MAGI so 20 000 iteracij × 200 korakov, polovica iteracij je burn-in, korak pa
se med burn-inom nastavi na 60–90 % sprejetih predlogov (MAGI, JSS §3). To je $4\cdot10^6$ izračunov
$\nabla\mathcal L$ na en model: odlično za en model in njegovo negotovost, za presejanje tisočih
kandidatov pa preveč.

### 3.9 Kaj MAGI ni

- ODE je izpolnjena **samo v točkah $I$**. Med njimi trajektorijo določa GP interpolacija. Pri ostrih
  spremembah v $f$ je treba mrežo zgostiti ali skrajšati $\phi_2$ (MAGI, JSS §4.2, opomba 3).
- MAGI priporoča zaporedje mrež $I_0\subset I_1\subset\cdots$, kjer je $I_0$ najmanjša enakomerna
  mreža, ki vsebuje $\tau$, vsaka naslednja pa ima med sosednjima točkama vrinjeno novo točko. Ocene
  naj bodo stabilne čez to zaporedje (MAGI, JSS §4.1).
- Numerične integracije ni nikjer. Je pa linearna algebra z matrikami $|I|\times|I|$.

---

## 4. Povezava z našim gradient matchingom in s seedi

**Trditev 4.1.** Naj bo $\tau\subset I$, naj bo prior MAGI enak GP-ju, s katerim gladimo podatke
(isti $\mathcal K_d$, $\mu_d$), in naj $\hat x_d$ označuje GP posteriorno povprečje iz podatkov.
Potem je pri $x_d(I)=\hat x_d(I)$ ODE ostanek MAGI natanko ostanek gradient matchinga:
$$
\dot\mu_d(I)+m_d\{\hat x_d(I)-\mu_d(I)\}=\hat x_d'(I).
$$

*Dokaz.* Posteriorno povprečje je $\hat x_d(t)=\mu_d(t)+\mathcal K_d(t,\tau)\alpha$ z
$\alpha=(\mathcal K_d(\tau,\tau)+\sigma_d^2 I)^{-1}(y_d-\mu_d(\tau))$, njegov odvod pa
$\hat x_d'(t)=\dot\mu_d(t)+{}'\mathcal K_d(t,\tau)\alpha$. Ker je $\tau\subset I$, je
$\mathcal K_d(I,\tau)=C_d E_\tau$, kjer $E_\tau$ izbere stolpce, ki pripadajo $\tau$. Zato
$C_d^{-1}\mathcal K_d(I,\tau)=E_\tau$ in
$$
m_d\{\hat x_d(I)-\mu_d(I)\}={}'\mathcal K_d(I,I)\,C_d^{-1}\mathcal K_d(I,\tau)\,\alpha
={}'\mathcal K_d(I,I)E_\tau\alpha={}'\mathcal K_d(I,\tau)\alpha=\hat x_d'(I)-\dot\mu_d(I).\qquad\blacksquare
$$

*Numerično preverjeno* z `GP1dim` (RBF, 25 meritev $\sin t$ s šumom, $I=\tau\,\cup$ sredine):
razlika je $1.2\cdot10^{-8}$ pri jitterju $10^{-10}\max\operatorname{diag}C$ in raste z jitterjem
(tabela v §6.7).

Posledice:

- **Najin gradient matching je MAGI-MAP z zamrznjenimi vrednostmi** $x(I)=\hat x(I)$, brez prior
  člena in z identiteto namesto uteži $\Psi_d^{-1}$. MAGI te vrednosti "odmrzne": opazovane
  komponente se smejo odmakniti od glajenja podatkov, kjer to zahteva ODE, neopazovane pa sploh
  postanejo mogoče.
- **Tvoja ideja s seedi je MAGI**: seedi so $x_\ell(I)$, interpolant skozi njih da $m_\ell$
  (Trditev 2.3), regularizacija je prior člen $\|x_\ell\|^2_{C_\ell^{-1}}$, utež "$\lambda$" pa
  ni prosta, ampak je $\Psi_\ell^{-1}/\beta$, določena z jedrom in mrežo.
- **Število seedov**: s prior členom več točk mreže ne pokvari rešitve, temveč le podraži linearno
  algebro, in MAGI ravno priporoča gostitev, dokler se ocene ne ustalijo.

---

## 5. Od Bayesa do optimizacije

### 5.1 MAP

$$
(\hat\theta,\hat x(I))=\arg\min_{\theta\in\Omega_\theta,\ x(I)}\ \mathcal L(\theta,x(I)),
$$
z $\mathcal L$ iz (3.2), pri fiksnih $\phi$ in $\sigma$. To je natanko korak 2 iz §3.7, brez
optimizacije $\phi$. **MAGI-MAP je MAGI brez HMC**.

### 5.2 Ali je to "maximum likelihood"?

Ne povsem. Ločiti moramo dve količini:

- **Marginalno verjetje** $\theta$ (pravi ML) integrira trajektorijo:
  $$\mathcal{M}(\theta)=\int p(y\mid x)\,p(x)\,p(\dot X(I)=f\mid x)\,dx(I).$$
  To je integral v $D|I|$ dimenzijah, ki ga ne znamo izračunati v zaprti obliki.
- **Skupno verjetje**, maksimizirano po $x$ (profil, v duhu h-likelihooda Leeja in Nelderja):
  $\ell_p(\theta)=-\min_{x}\mathcal L(\theta,x)$. Pri ravnem priorju na $\theta$ je MAP $\hat\theta$
  ravno $\arg\max\ell_p$.

Povezuje ju **Laplaceova aproksimacija** (Tierney in Kadane 1986). Naj bo
$\hat x(\theta)=\arg\min_x\mathcal L(\theta,x)$ in $H(\theta)=\nabla_x^2\mathcal L(\theta,\hat x(\theta))\succ0$:
$$
\log\mathcal M(\theta)\approx \ell_p(\theta)+\frac{D|I|}{2}\log 2\pi-\frac12\log\det H(\theta).
\tag{5.1}
$$

- Za **ocenjevanje $\theta$** je razlika majhna, kadar je $H(\theta)$ šibko odvisna od $\theta$
  (podatki dobro določajo $x$). Pri neopazovanih komponentah to ne drži nujno.
- Za **primerjavo struktur** (različni $f$) je člen $\log\det H$ različen med kandidati in deluje kot
  Occamov faktor. Za primerjavo kandidatov z različnim številom parametrov bi rabili še Laplace po
  $\theta$, s pravim (normaliziranim) priorjem, npr. enakomernim na `range` konstant.
- Zaradi temperiranja (§3.5) $\mathcal M$ ni prava gostota podatkov, zato je to vrednost za
  razvrščanje, ne za statistični test.

### 5.3 Zapis kot nelinearni najmanjši kvadrati

Za $A=LL^\top$ (Cholesky) velja $\|v\|^2_{A^{-1}}=\|L^{-1}v\|^2$. Zato je
$\mathcal L=\tfrac12\|R(z)\|^2+\text{konst.}-\log\pi(\theta)$, $z=(x(I),\theta)\in\mathbb R^{D|I|+p}$,
z ostanki:
$$
\begin{aligned}
R^{\text{pod}}_d &= \big(x_d(\tau_d)-y_d\big)/\sigma_d &&\in\mathbb R^{|\tau_d|},\\
R^{\text{prior}}_d &= \beta^{-1/2}\,L_{C_d}^{-1}\big(x_d(I)-\mu_d(I)\big) &&\in\mathbb R^{|I|},\\
R^{\text{ODE}}_d &= \beta^{-1/2}\,L_{\Psi_d}^{-1}\big(f_d(x(I),\theta,t_I)-\dot\mu_d(I)-m_d\{x_d(I)-\mu_d(I)\}\big) &&\in\mathbb R^{|I|}.
\end{aligned}
$$
Raven prior na `range` konstant pomeni meje na $\theta$, kar `scipy.optimize.least_squares(method="trf")`
podpira neposredno. Podatkovni in prior ostanki so linearni v $x$, nelinearen je samo ODE ostanek,
prek $f$. Če je $f$ linearen v $x$, je pri fiksnem $\theta$ problem po $x$ linearen in ima zaprto
rešitev.

### 5.4 Degeneracije MAP

Bayesovsko vzorčenje gleda **maso** posteriorja, MAP pa njegov **vrh**. Pri modelih z latentnimi
spremenljivkami ima lahko gostota neskončen vrh na mestu, kjer je mase malo. Spodnje tri trditve
povedo, česa MAP ne sme optimizirati.

**Trditev 5.1 (šum).** Če $\sigma_d$ opazovane komponente optimiziramo skupaj z $x$, je
$\inf\mathcal L=-\infty$.

*Dokaz.* Postavimo $x_d(\tau_d)=y_d$. Podatkovni del $\mathcal L$ je potem
$|\tau_d|\log\sigma_d+\text{konst.}\to-\infty$ za $\sigma_d\to0$, ostali členi pa so končni. $\blacksquare$

$\sigma_d$ zato fiksiramo iz GP fita (korak 1). MAGI $\sigma$ sme vzorčiti, ker je masa okoli
$\sigma\to 0$ končna (integral verjetja po $x_d(\tau_d)$ je 1), čeprav je gostota tam neomejena.

**Trditev 5.2 (varianca latentnega priorja).** Naj bo $\mu_\ell=0$ in naj obstajata $\theta^*\in\Omega_\theta$
in opazovani del $x_o(I)$, pri katerih $x_\ell\equiv0$ reši latentno enačbo,
$f_\ell(x_o(I),0,\theta^*,t_I)=0$. Če varianco $\phi_{\ell,1}$ optimiziramo skupaj z $x$, je
$\inf\mathcal L=-\infty$.

*Dokaz.* Jedro je linearno v $\phi_1$, zato $C_\ell=\phi_1\tilde C_\ell$, $\Psi_\ell=\phi_1\tilde\Psi_\ell$,
$m_\ell$ pa od $\phi_1$ ni odvisen. Pri $x_\ell=0$ in $\theta=\theta^*$ izgineta obe latentni kvadratni
formi. Od latentnega bloka ostane $\tfrac1{2\beta}(\log\det C_\ell+\log\det\Psi_\ell)=\tfrac{|I|}{\beta}\log\phi_1+\text{konst.}\to-\infty$
za $\phi_1\to0$, preostanek pa je končen. $\blacksquare$

*Primer:* Lotka-Volterra s skrito $x_2$: $f_2=-c\,x_2+d\,x_1x_2$ je pri $x_2\equiv0$ enak 0 za vsak
$\theta$. MAGI v koraku 2 $\phi_\ell$ optimizira. Ali v praksi pade v to degeneracijo, je odvisno
od začetka in v članku ni obravnavano. Pri MAP $\phi_\ell$ fiksiramo ali omejimo navzdol.

**Trditev 5.3 (simetrija skale).** Naj bo $\mu_\ell=0$ in naj za $\lambda\in(0,1]$ obstaja preslikava
$\theta\mapsto\theta_\lambda\in\Omega_\theta$, za katero velja za vse $x,t$:
$$
f_o(x_o,\lambda x_\ell,\theta_\lambda,t)=f_o(x_o,x_\ell,\theta,t),\qquad
f_\ell(x_o,\lambda x_\ell,\theta_\lambda,t)=\lambda\,f_\ell(x_o,x_\ell,\theta,t).
$$
Potem je $\mathcal L(\theta_\lambda,x_o,\lambda x_\ell)=A+\lambda^2B$, kjer $A$ in $B\ge0$ nista odvisna
od $\lambda$, in $B>0$ za $x_\ell\neq0$. Minimum zato ni dosežen pri $x_\ell\ne0$: MAP drsi proti
$\lambda\to0$, torej $x_\ell\to0$, $\theta_\lambda$ pa proti robu $\Omega_\theta$.

*Dokaz.* Podatkovni in prior člen opazovanih komponent $x_\ell$ ne vsebujeta. ODE člen opazovanih
komponent vsebuje $f_o$, ki je po predpostavki invarianten. Latentni prior je $\lambda^2\|x_\ell\|^2_{C_\ell^{-1}}$,
latentni ODE ostanek pa $f_\ell(\cdot,\lambda x_\ell,\theta_\lambda)-m_\ell\lambda x_\ell=\lambda(f_\ell-m_\ell x_\ell)$.
Torej je $B=\tfrac1{2\beta}\big(\|x_\ell\|^2_{C_\ell^{-1}}+\|f_\ell-m_\ell x_\ell\|^2_{\Psi_\ell^{-1}}\big)$,
ki je pozitiven, ker je $C_\ell\succ0$. $\blacksquare$

*Primer:* Lotka-Volterra s skrito $x_2$: $\theta_\lambda=(a,\,b/\lambda,\,c,\,d)$. To je ista
neidentifikabilnost, ki smo jo videli pri `ms` (določen je le $b\cdot x_2$). Pri `ms` je bila smer
**ravna**, pri MAP pa **pada** proti robu, ker prior nagrajuje majhen $x_\ell$.

*Kaj storiti:* simetrijo fiksiramo, preden fitamo. Možnosti so fiksirati eno konstanto (npr. $b$),
fiksirati $x_\ell(t_0)$ na znano vrednost, ali $\mu_\ell$ postaviti na verjetno raven. Pri watershedu
je podobna (približna) simetrija premik: `uszDelta` $+\delta$ in `soilWaterCapacity` $+\delta$
dajeta isto perkolacijo, dokler omejitev ET ne zagrabi. Prior z $\mu=0$ bi oboje vlekel navzdol.

### 5.5 Negotovost brez HMC

Gauss-Newtonov Hessian $J^\top J$ v MAP točki da Laplaceovo aproksimacijo posteriorja,
$z\mid y\approx\mathcal N(\hat z,(J^\top J)^{-1})$. Ustrezni blok da približne intervale za
$\theta$, kar je praktično zastonj, ker $J$ že imamo. Velja pa le, če je posterior približno
Gaussovski in nobena meja ni aktivna. Zaradi temperiranja so to intervali temperiranega
posteriorja. Taki intervali bi lahko nadomestili najin interval zaupanja pri rezanju kandidatov.

---

## 6. Implementacija v `odestimate`

### 6.1 Vmesnik

Po vzoru `latent.multishooting.ms`, da ga PyBM adapter kliče na enak način:

```python
def magi_map(f, ts, xs, theta_0, unobserved=None, grid=None, gps=None, latent_kernels=None,
             mu=None, beta=None, bounds=None, engine="torch", unobserved_inits=None,
             **least_squares_kwargs) -> OptimizeResult:
    """xs: (n, D), stolpci v `unobserved` so NaN. grid: I (privzeto najmanjša enakomerna mreža, ki
    vsebuje ts). gps: fitani GP1dim za opazovane komponente (sicer fit tukaj).
    latent_kernels: fiksna jedra za neopazovane komponente.
    Vrne .theta, .x (|I|, D), .grid, .beta in .cov (Gauss-Newton, §5.5)."""
```

PyBM adapter `pybm/estimate/magi.py` je analogen `multishooting.py`: stanja s podatki so opazovana,
ostala latentna, `build_problem` da $f$ samo s konstantami, ki v enačbah nastopajo.

### 6.2 Predračun

Odvisen je samo od mreže $I$ in jeder, **ne od kandidata in ne od $\theta$**. Izračuna se enkrat in
deli med vsemi kandidati, kot `precompute` pri gradient matchingu. Za vsako komponento $d$:

```
C   = k(I, I) + eps_C * I                 # k: fitano jedro, v originalnih enotah, BREZ šuma:
                                          # RBFWithNoise: k(I[:, None], I[:, None]) z eksplicitnim Y,
                                          # k(I[:, None]) bi na diagonalo prištel noise_var
L_C = cholesky(C)
dK  = k.gradient_X(I, I)                  # 'K(I,I)
ddK = k.hessian_XY(I, I)                  # K''(I,I)
m   = cho_solve(L_C, dK.T).T              # 'K C^-1
Psi = ddK - m @ dK.T + eps_Psi * I        # K' = 'K^T
L_P = cholesky(Psi)
shrani: inv(L_C), inv(L_P), inv(L_P) @ m  # za ostanke in Jacobian
```

**Enote:** `GP1dim` z `normalize_y=True` fita jedro na standardiziranih podatkih. Prior v originalnih
enotah je potem $s^2\,k$, $\mu=\bar y$, kjer sta $s$ in $\bar y$ `_y_train_std` in `_y_train_mean`.

### 6.3 Ostanki in Jacobian

Neznanke so $z=(x_1(I),\dots,x_D(I),\theta)$. Ostanki so iz §5.3. Jacobian po blokih, kjer je
$E_{\tau_d}$ izbirna matrika:
$$
\begin{aligned}
\partial R^{\text{pod}}_d/\partial x_d &= E_{\tau_d}^\top/\sigma_d, \\
\partial R^{\text{prior}}_d/\partial x_d &= \beta^{-1/2}L_{C_d}^{-1}, \\
\partial R^{\text{ODE}}_d/\partial x_e &= \beta^{-1/2}L_{\Psi_d}^{-1}\Big(\operatorname{diag}\big(\tfrac{\partial f_d}{\partial x_e}(t_i)\big)_{i\in I}-\delta_{de}\,m_d\Big), \\
\partial R^{\text{ODE}}_d/\partial\theta &= \beta^{-1/2}L_{\Psi_d}^{-1}\,\partial f_d(x(I),\theta,t_I)/\partial\theta .
\end{aligned}
$$
$f$ deluje po točkah, zato je $\partial f_d(t_i)/\partial x_e(t_j)=\delta_{ij}\,(F_x(t_i))_{de}$. Vse
$F_x(t_i)$ dobimo z $D$ klici `jvp` (torch) ali $D$ batch končnimi diferencami (numpy), enako kot v
`confidence_interval` in `ms`. $L^{-1}\operatorname{diag}(v)$ je $L^{-1}$ s skaliranimi stolpci, kar
stane $O(|I|^2)$. Rešuje `least_squares(method="trf", x_scale="jac", bounds=...)`.

### 6.4 Začetne vrednosti

Dve stopnji:

- **A:** $x_o(I):=\hat x_o(I)$ zamrznemo (GP povprečje), optimiziramo $(\theta,x_\ell(I))$. To je tvoja
  ideja s seedi. Brez latentnih komponent je to po Trditvi 4.1 najin gradient matching z utežmi
  $\Psi^{-1}$. Začetni $x_\ell(I)$ je konstanta (`unobserved_inits`), ali pa ena poceni integracija
  *samo latentnih* enačb s fiksnim $x_o=\hat x_o$. Ta je stabilnejša od integracije celega sistema,
  ker opazovane komponente držijo podatki.
- **B:** sprostimo vse $x(I)$ in $\theta$ in začnemo iz rešitve A.

Kot dodatna robustnost (naša ideja, ne MAGI): kontinuacija po $\beta$. Začnemo z večjim $\beta$
(šibkejša GP člena) in ga zmanjšujemo do priporočene vrednosti, enako kot homotopija.

### 6.5 Hiperparametri v praksi

- **Opazovane komponente:** `GP1dim` na podatkih. Preveri fit: degeneracijo $\ell\to0$ smo že videli,
  npr. z `normalize_y=True` na 25 točkah $\sin t$. $\sigma_d$ fiksiran (Trditev 5.1).
- **Latentne komponente:** $\ell_\ell$ izposojen od opazovane komponente, s katero je sklopljena,
  $\phi_{\ell,1}$ iz pričakovane velikosti, $\mu_\ell$ verjetna raven. Vse fiksno (Trditvi 5.2 in 5.3).
- $\beta=D|I|/N$ (§3.5).
- Simetrije modela odpravi, preden fitaš (Trditev 5.3).

### 6.6 Jedra

V `odestimate/gp/kernels.py` je samo `RBFWithNoise` (z `gradient_X` in `hessian_XY`). Za MAGI dodamo
Matérn 3/2 in 5/2 v zaprti obliki. Splošni $\nu$ (npr. 2.01) bi rabil Besslove funkcije
(`scipy.special.kv`). Za stacionarno jedro velja ${}'\mathcal K=\kappa'(s-t)$ in $\mathcal K''=-\kappa''(s-t)$.

Matérn 3/2, $a=\sqrt3/\ell$, $u=a|r|$ (proces je enkrat odvedljiv, $\nu=3/2>1$):
$$
\kappa=\sigma^2(1+u)e^{-u},\qquad \kappa'=-\sigma^2a^2\,r\,e^{-u},\qquad -\kappa''=\sigma^2a^2(1-u)e^{-u}.
$$
Matérn 5/2, $a=\sqrt5/\ell$, $u=a|r|$:
$$
\kappa=\sigma^2\big(1+u+\tfrac{u^2}{3}\big)e^{-u},\qquad
\kappa'=-\sigma^2\tfrac{a^2 r}{3}(1+u)e^{-u},\qquad
-\kappa''=\sigma^2\tfrac{a^2}{3}\big(1+u-u^2\big)e^{-u}.
$$
*Numerično preverjeno* s končnimi diferencami: napaka $\kappa'$ je $\sim10^{-10}$, napaka $\kappa''$
$\sim10^{-6}$ (red diferenčne sheme). Velja tudi $\operatorname{Var}\dot X=-\kappa''(0)$, torej
$3\sigma^2/\ell^2$ oziroma $5\sigma^2/(3\ell^2)$.

### 6.7 Numerika

**Pogojenost.** Pri RBF z $\ell\approx2$ na 49 točkah v $[0,10]$ smo izmerili
$\operatorname{cond}(C)=1.7\cdot10^{19}$. Cholesky brez dodatka na diagonali odpove. Dodatek
$\varepsilon\max\operatorname{diag}C$ to reši, pokvari pa identiteto iz Trditve 4.1:

| $\varepsilon$ | $10^{-10}$ | $10^{-8}$ | $10^{-6}$ | $10^{-4}$ |
|---|---|---|---|---|
| $\max\lvert m\hat x-\hat x'\rvert$ | $1.2\cdot10^{-8}$ | $9.1\cdot10^{-7}$ | $1.2\cdot10^{-4}$ | $5.0\cdot10^{-3}$ |

(Velikost $\hat x'$ je reda 1.) $\Psi_d$ je pogojena še slabše (Trditev 2.4). Manj gladka jedra
(Matérn) so bolje pogojena, kar je dodaten razlog zanje.

**Velike mreže.** Za $|I|$ v tisočih (watershed: 2557 dni) MAGI aproksimira $C_d^{-1}$, $\Psi_d^{-1}$ in
$m_d$ s pasovnimi matrikami (`bandSize = 20`, JSS §4.2). Za nas to pomeni: Cholesky pasovne
natančnostne matrike je pasoven, Jacobian je zato redek, kar omogoči
`least_squares(tr_solver="lsmr", jac_sparsity=...)`. Pasovna aproksimacija inverza pa ni nujno
pozitivno definitna, kar MAGI tudi zazna in opozori.

### 6.8 Cena

| | `ms` (multiple shooting) | MAGI-MAP |
|---|---|---|
| neznanke | $n_{\text{sub}}D+p$ | $D\lvert I\rvert+p$ |
| ODE izpolnjena | natančno (do integratorja), na celem intervalu | mehko, v točkah $I$ |
| integracija | da, vsaka iteracija | ne |
| na iteracijo | integracije $\times(1+D+p)$ kopij | $f$, $F_x$, $F_\theta$ v $\lvert I\rvert$ točkah; $O(D^2\lvert I\rvert^2)$ za Jacobian; $O((D\lvert I\rvert)^3)$ za korak (gosto) |
| enkratno | – | $O(D\lvert I\rvert^3)$ predračun, deljen med kandidati |
| razpad pri slabem $\theta$ | da (`int_threshold`) | ne |

Pri $|I|\lesssim 500$ je gosta varianta preprosta in hitra, nad tem rabimo pasovno.

### 6.9 Testni načrt

1. **Trditev 4.1:** brez latentnih komponent in z zamrznjenim $x$ mora MAGI-MAP dati isti $\theta$
   kot `gradient_matching` z utežmi $\Psi^{-1}$. Identiteto ostankov smo že preverili.
2. **Lotka-Volterra s skrito $x_2$:** brez fiksiranja simetrije mora $b$ zdrsniti k meji
   (Trditev 5.3). S fiksnim $b$ (ali primerjavo $b\cdot x_2(0)$) primerjamo z `ms`.
3. **Hes1 s skrito komponento $H$:** glavni primer MAGI (JSS §3). Podatke generiramo sami.
4. **Watershed s skritim `uszDelta`:** primerjava z `ms` ($cn\approx60$, `coverCoeff`
   $\approx0.8$, $rc+sc\approx0.15$) po rezultatu in času.
5. **Zaporedje mrež** $I_0\subset I_1\subset I_2$: ocene naj se ustalijo.

### 6.10 Odprta vprašanja

- **Merilo za razvrščanje kandidatov:** $\ell_p(\hat\theta)$ ali Laplaceov (5.1) (§5.2)? Pri
  neopazovanih komponentah se lahko razlikujeta.
- **Opazovanje funkcije stanj.** Ker so vrednosti $x(\tau)$ eksplicitne neznanke, se meritev
  algebraične spremenljivke (npr. pretok, $y=g(x,\theta,t)+\varepsilon$) vključi preprosto:
  $R^{\text{pod}}=(g(x(\tau),\theta,\tau)-y)/\sigma$, Jacobian po verižnem pravilu. To je pri MAGI
  bistveno lažje kot pri `ms`, in prav to rabimo za watershed. Vsa stanja so takrat latentna, zato
  vprašanje hiperparametrov iz §6.5 velja za vse.
- **Diskretni modeli.** Watershed (GWLF) je v resnici dnevna diferenčna enačba. Pri njej bi bil
  pravilnejši pogoj $x_{i+1}-x_i=h\,f(x_i)$ namesto $\dot x=f$. Ta "diferenčni MAGI" ne potrebuje
  odvoda GP-ja in je blizu šibko vezanemu 4D-Var oziroma glajenju stanj. Vredno bi ga bilo
  primerjati.
- **Neposredna uporaba MAGI kot reference.** Python verzija zahteva prevajanje iz izvorne kode
  (`build.sh`, `pymagi/py_build.sh`) in Jacobiana `fOdeDx` ter `fOdeDtheta`. Te lahko zgradimo iz
  PyBM torch $f$ s `torch.func`.

---

## 7. Viri

- S. Yang, S. W. K. Wong, S. C. Kou (2021), *Inference of dynamic systems from noisy and sparse data
  via manifold-constrained Gaussian processes*, PNAS 118(15), e2020397118.
  [doi:10.1073/pnas.2020397118](https://www.pnas.org/doi/10.1073/pnas.2020397118)
- S. W. K. Wong, S. Yang, S. C. Kou (2024), *magi: A Package for Inference of Dynamic Systems from
  Noisy and Sparse Data via Manifold-Constrained Gaussian Processes*, J. Stat. Softw. 109(4).
  [doi:10.18637/jss.v109.i04](https://doi.org/10.18637/jss.v109.i04),
  [arXiv:2203.06066](https://arxiv.org/abs/2203.06066). Iz tega članka so formule v §3 in §6.7.
- Koda: [github.com/wongswk/magi](https://github.com/wongswk/magi) (R, MATLAB, Python).
- C. E. Rasmussen, C. K. I. Williams (2006), *Gaussian Processes for Machine Learning*, MIT Press,
  §4.2.1 (Matérn, odvedljivost), §9.4 (odvodi GP-jev).
- E. Solak et al. (2003), *Derivative observations in Gaussian process models of dynamic systems*,
  NeurIPS 16.
- J. O. Ramsay, G. Hooker, D. Campbell, J. Cao (2007), *Parameter estimation for differential
  equations: a generalized smoothing approach*, JRSS-B 69(5), 741–796. Isti problem kot penalizirana
  regresija z B-zlepki.
- P. Wenk et al. (2020), *ODIN: ODE-Informed Regression for Parameter and State Inference in
  Time-Continuous Dynamical Systems*, AAAI 34(04). [arXiv:1902.06278](https://arxiv.org/abs/1902.06278)
- Y. Lee, J. A. Nelder (1996), *Hierarchical generalized linear models*, JRSS-B 58(4), 619–678
  (h-likelihood).
- L. Tierney, J. B. Kadane (1986), *Accurate approximations for posterior moments and marginal
  densities*, JASA 81, 82–86 (Laplaceova aproksimacija).
- R. M. Neal (2011), *MCMC using Hamiltonian dynamics*, v *Handbook of Markov Chain Monte Carlo*.
