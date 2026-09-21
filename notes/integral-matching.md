# Integral matching: matematične osnove

*Analiza napake, identifikabilnosti in primerjava z gradient matchingom pri ocenjevanju parametrov ODE iz zašumljenih podatkov.*

---

## Kako brati ta učbenik

Vsako poglavje uvede en ali dva nova pojma, ju formalno definira, poda vir za nadaljnje branje, nato pa pojem takoj uporabi v izpeljavi. Izpeljave so zapisane po korakih — brez preskokov. Kjer trditev ni strogo dokazana, ampak je hevristični (dimenzijski) argument, je to eksplicitno označeno.

Notacija je fiksna skozi cel dokument:

- $x(t)\in\mathbb R^d$ — prava (neznana) rešitev ODE,
- $c\in\mathbb R^p$ — vektor parametrov, $c_0$ njegova prava vrednost,
- $F:\mathbb R^d\times\mathbb R\times\mathbb R^p\to\mathbb R^d$ — desna stran enačbe, $x'=F(x,t,c)$,
- $y_i = x(t_i) + \eta_i$, $\eta_i\sim$ šum — opazovani podatki,
- $\hat x(t)$ — interpolant (GP posteriorna povprečna funkcija), ocenjen iz $\{(t_i,y_i)\}$,
- $t_0<t_1<\dots<t_n$ — mreža časov, $\Delta_i = t_i-t_{i-1}$, $T=t_n-t_0$.

---

## 1. Gradniki

### 1.1 Problem ocenjevanja parametrov ODE

**Definicija 1.1.** Dana je družina ODE $x'=F(x,t,c)$, $c\in\Theta\subset\mathbb R^p$, in zašumljena opazovanja $y_i=x(t_i)+\eta_i$ prave rešitve $x(\cdot)$, ki reši sistem pri neznanem $c=c_0$. Cilj je oceniti $c_0$ iz $\{y_i\}$.

To je poseben primer *inverznega problema* — iščemo vzrok (parameter) iz posledice (zašumljena opazovanja), kar je po definiciji Hadamarda težje kot direkten problem, ker rešitev ni nujno zvezno odvisna od podatkov.

> **Vir:** J. Hadamard, *Sur les problèmes aux dérivées partielles et leur signification physique*, Princeton University Bulletin 13 (1902), 49–52 — izvor pojma dobro/slabo postavljenega problema (uporabimo ga v poglavju 6).

### 1.2 Gaussovski proces kot interpolant

**Definicija 1.2 (Gaussovski proces).** $\hat x$ je Gaussovski proces (GP), če je za vsak končen nabor točk $t_1,\dots,t_m$ vektor $(\hat x(t_1),\dots,\hat x(t_m))$ skupno normalno porazdeljen. GP je določen s povprečno funkcijo $\mu(t)$ in kovariančnim jedrom $k(t,t')$.

Pri regresiji z opazovanji $y_i=x(t_i)+\eta_i$, $\eta_i\sim\mathcal N(0,\sigma^2)$ neodvisno, je posteriorna povprečna funkcija (ki jo vzamemo za interpolant $\hat x$) enaka

$$
\hat x(t) = k(t)^\top (K+\sigma^2 I)^{-1} y, \qquad K_{ij}=k(t_i,t_j),\quad k(t)_i = k(t,t_i). \tag{1.1}
$$

Za stacionarno jedro (npr. kvadratno eksponentno, squared-exponential)

$$
k(t,t') = \sigma_f^2\exp\!\left(-\frac{(t-t')^2}{2\ell^2}\right) \tag{1.2}
$$

parameter $\ell$ imenujemo **korelacijska dolžina (lengthscale)** — razdalja, na kateri jedro pade na $\sim e^{-1/2}$ svoje vrednosti; to je "karakteristična skala gladkosti" funkcije, ki jo GP modelira. Ta $\ell$ bo ključna v poglavju 5.

> **Vir:** C. E. Rasmussen, C. K. I. Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006 — poglavje 2 za regresijsko formulo (1.1), poglavje 4.2 za lengthscale in stacionarna jedra.

### 1.3 Lipschitzova zveznost

**Definicija 1.3.** Funkcija $F$ je *$L_x$-Lipschitzova v $x$* (enakomerno v $t,c$), če

$$
\|F(x,t,c)-F(x',t,c)\| \le L_x\|x-x'\| \qquad \forall x,x',t,c. \tag{1.3}
$$

Analogno definiramo $L_c$-Lipschitzovost v $c$. Ta pogoj je standarden pogoj za eksistenco in enoličnost rešitve ODE (Picard–Lindelöf) in bo v naslednjih poglavjih edino, kar potrebujemo o $F$, da lahko kontroliramo, kako se napaka v $\hat x$ prenese v napako v $c$.

> **Vir:** W. Rudin, *Principles of Mathematical Analysis*, 3. izd., McGraw-Hill, 1976, §9 — definicija; E. A. Coddington, N. Levinson, *Theory of Ordinary Differential Equations*, McGraw-Hill, 1955, pogl. 1 — vloga Lipschitzovosti pri ODE.

### 1.4 Enakomerna napaka interpolacije

**Definicija 1.4.** Naj bo $e(t):=\hat x(t)-x(t)$ napaka interpolanta. Predpostavimo, da obstaja $\varepsilon>0$, tako da z visoko verjetnostjo

$$
\sup_t \|e(t)\| \le \varepsilon. \tag{1.4}
$$

Konstanto $\varepsilon$ v tem dokumentu jemljemo kot **dano** (eksogeno) — v praksi jo dobimo bodisi iz širine GP-jevega kredibilnega pasu (posteriorna standardna deviacija $\times$ kvantil), bodisi iz teoretičnih rezultatov o **posteriorni kontrakciji** GP regresije, ki povedo, s kakšno hitrostjo $\varepsilon\to0$, ko število podatkov $\to\infty$, glede na gladkost prave funkcije in izbiro jedra.

> **Vir:** A. W. van der Vaart, J. H. van Zanten, *Rates of contraction of posterior distributions based on Gaussian process priors*, Annals of Statistics 36(3), 2008, 1435–1463.

### 1.5 RKHS in odvod GP

Interpolant $\hat x$ ni le zvezen, ampak (za gladka jedra kot (1.2)) neskončnokrat odvedljiv, in obstaja natančna povezava med gladkostjo funkcije v pripadajočem **reproducirajočem jedrnem Hilbertovem prostoru (RKHS)** in tem, kako se napaka $\varepsilon$ ojači pri odvajanju. To bomo potrebovali v poglavju 6; za zdaj samo definicija.

**Definicija 1.5.** RKHS $\mathcal H_k$, pridružen jedru $k$, je Hilbertov prostor funkcij, v katerem velja *reproducing property* $f(t)=\langle f,k(\cdot,t)\rangle_{\mathcal H_k}$. Norma $\|f\|_{\mathcal H_k}$ meri "gladkost" funkcije glede na $k$.

> **Vir:** G. Wahba, *Spline Models for Observational Data*, SIAM, 1990, pogl. 1 — RKHS; Rasmussen & Williams (2006), pogl. 9.4 — odvodi GP in njihova (ko)varianca.

---

## 2. Metoda integral matching

### 2.1 Definicija izgube

**Definicija 2.1 (Integral matching loss).** Za dano mrežo $t_0<\dots<t_n$ in kandidatni parameter $c$ definiramo residuale

$$
l_i(c) := \hat x(t_i)-\hat x(t_{i-1}) - \int_{t_{i-1}}^{t_i} F(\hat x(\tau),\tau,c)\,d\tau, \qquad i=1,\dots,n, \tag{2.1}
$$

in skupno izgubo $l(c)=\sum_i \|l_i(c)\|$ (ali $J(c)=\sum_i\|l_i(c)\|^2$ za gladko optimizacijo). Ocena parametra je $\hat c=\arg\min_c J(c)$.

Intuitivno: (2.1) preverja, ali interpolant *lokalno reši* ODE v integralski (šibki) obliki na vsakem podintervalu — namesto da bi zahtevali $\hat x'(t)\approx F(\hat x,t,c)$ v vsaki točki (to je *gradient matching*, poglavje 6), zahtevamo ujemanje po *integriranem* prirastku.

### 2.2 Proces napake $e(t)$

Ker $x$ reši ODE pri $c_0$, velja $x(t)=x(t_0)+\int_{t_0}^t F(x(\tau),\tau,c_0)\,d\tau$. Po definiciji $e(t)=\hat x(t)-x(t)$ torej

$$
e(t) = \hat x(t) - x(t_0) - \int_{t_0}^t F(x(\tau),\tau,c_0)\,d\tau. \tag{2.2}
$$

Iz (2.2), za dva zaporedna časa:

$$
e(t_i)-e(t_{i-1}) = \big[\hat x(t_i)-\hat x(t_{i-1})\big] - \int_{t_{i-1}}^{t_i} F(x(\tau),\tau,c_0)\,d\tau. \tag{2.3}
$$

### 2.3 Razčlenitev $l_i(c)$

Iz (2.3) izrazimo $\hat x(t_i)-\hat x(t_{i-1}) = e(t_i)-e(t_{i-1}) + \int_{t_{i-1}}^{t_i}F(x,\tau,c_0)\,d\tau$ in vstavimo v (2.1):

$$
\boxed{\; l_i(c) = \underbrace{e(t_i)-e(t_{i-1})}_{\text{interpolacijski šum}} \;+\; \underbrace{\int_{t_{i-1}}^{t_i}\Big[F(x(\tau),\tau,c_0)-F(\hat x(\tau),\tau,c)\Big]d\tau}_{\text{signal o } c} \;} \tag{2.4}
$$

To je ključna formula tega dokumenta: pove, da je vsak residual vsota (a) šuma, ki izvira izključno iz napake interpolanta, in (b) člena, ki nosi informacijo o razliki $c-c_0$. Vsa naslednja poglavja so posledice te enačbe.

---

## 3. Meja za izgubo pri $c_0$ in kriterij ustavitve

### 3.1 Groba (worst-case) meja

Znotraj integrala v (2.4) prištejemo in odštejemo $F(\hat x,\tau,c_0)$:

$$
F(x,\tau,c_0)-F(\hat x,\tau,c) = \underbrace{\big[F(x,\tau,c_0)-F(\hat x,\tau,c_0)\big]}_{\le L_x\|e(\tau)\| \text{ po (1.3)}} + \underbrace{\big[F(\hat x,\tau,c_0)-F(\hat x,\tau,c)\big]}_{\le L_c\|c-c_0\|}. \tag{3.1}
$$

Z (1.4) in trikotniško neenakostjo iz (2.4) in (3.1):

$$
\|l_i(c)\| \le 2\varepsilon + L_x\varepsilon\Delta_i + L_c\Delta_i\|c-c_0\|. \tag{3.2}
$$

Pri $c=c_0$ zadnji člen odpade. Seštejemo po $i=1,\dots,n$ (upoštevamo $\sum_i\Delta_i=T$):

$$
\boxed{\; \|l(c_0)\|_1 = \sum_{i=1}^n\|l_i(c_0)\| \;\le\; 2n\varepsilon + L_x\varepsilon T \;} \tag{3.3}
$$

**Interpretacija.** Izguba pri *pravem* parametru ni nič — spodaj je omejena z ravnjo šuma $O(n\varepsilon)$. Optimizacija pod to mejo je fitanje šuma v interpolantu, ne iskanje boljšega $c$. To je naravni **kriterij ustavitve**.

### 3.2 Zakaj je (3.3) preohlapna

Meja (3.3) sešteva $n$ absolutnih vrednosti prek trikotniške neenakosti, kar je pesimistično, če so $e(t_i)-e(t_{i-1})$ približno neodvisne naključne spremenljivke s povprečjem nič (kar velja, kadar je $\Delta_i$ velik v primerjavi z lengthscale $\ell$ — glej poglavje 5). V tem primeru za $J(c_0)=\sum_i\|l_i(c_0)\|^2$ **koncentracijske neenakosti** (npr. Bernsteinova) povedo, da $J(c_0)$ z visoko verjetnostjo leži blizu svoje pričakovane vrednosti $O(n\varepsilon^2)$, z nihanji reda $O(\sqrt n\,\varepsilon^2)$ — bistveno ostrejše kot $O(n^2\varepsilon^2)$, ki bi ga dala naivna kvadrirana (3.3).

> **Vir:** S. Boucheron, G. Lugosi, P. Massart, *Concentration Inequalities: A Nonasymptotic Theory of Independence*, Oxford University Press, 2013, pogl. 2 (Hoeffding) in pogl. 2.8 (Bernstein).

### 3.3 Praktičen kriterij ustavitve

Ker $L_x$ redko poznamo natančno, analitična meja (3.3) v praksi služi le kot red velikosti. Bolj zanesljiv pristop je **parametrični bootstrap**: iz GP posteriorja vzorčimo realizacije šuma, preračunamo $J$ pri trenutnem $\hat c$, in dobimo empirično ničelno porazdelitev $J(c_0)$. Ustavimo optimizacijo, ko $J(\hat c)$ pade v to porazdelitev (analogno $\chi^2$-testu dobrega prileganja z $\approx nd$ prostostnimi stopnjami). To je natanko logika ustavitvenega kriterija v metodi *generalized profiling / parameter cascading*.

> **Vir:** J. O. Ramsay, G. Hooker, D. Campbell, J. Cao, *Parameter estimation for differential equations: a generalized smoothing approach*, JRSS-B 69(5), 2007, 741–796 — §3–4 za kriterij ujemanja glajenja s podatki.

---

## 4. Identifikabilnost in radij nerazločljivosti

### 4.1 Fisherjeva informacijska matrika

**Definicija 4.1.** Za statistični model z gladko log-verjetjem je Fisherjeva informacijska matrika $I(c)=\mathbb E[\nabla_c\log p(y;c)\nabla_c\log p(y;c)^\top]$; njen inverz je (asimptotično) spodnja meja za kovarianco nepristranske cenilke (Cramér–Rao). Pri nelinearni regresiji z Gauss-Newtonovo aproksimacijo igra analogno vlogo matrika $J_r^\top J_r$, kjer je $J_r$ jakobijan residualov.

> **Vir:** G. A. F. Seber, C. J. Wild, *Nonlinear Regression*, Wiley, 2003, pogl. 2 (asimptotična teorija) in pogl. 12 (Fisherjeva informacija za ODE modele).

### 4.2 Gauss-Newton linearizacija in matrika $G$

Predpostavimo, da smo že blizu dna šuma ($J(c_0)$ majhen, poglavje 3), in lineariziramo $l_i(c)$ okoli $c_0$ z Definicijo 1.5-tipa odvodom $\partial F/\partial c$:

$$
l_i(c) \approx l_i(c_0) - S_i\,(c-c_0), \qquad S_i := \int_{t_{i-1}}^{t_i}\frac{\partial F}{\partial c}\big(\hat x(\tau),\tau,c_0\big)\,d\tau \;\in\mathbb R^{d\times p}. \tag{4.1}
$$

$S_i$ imenujemo **integrirana matrika občutljivosti** na $i$-tem intervalu. Kvadratna izguba se razvije kot

$$
J(c) \approx J(c_0) - 2(c-c_0)^\top\!\sum_i S_i^\top l_i(c_0) + (c-c_0)^\top G\,(c-c_0), \qquad \boxed{G:=\sum_{i=1}^n S_i^\top S_i.} \tag{4.2}
$$

$G$ je tu natanko Gauss-Newtonov Hessian — analog Fisherjeve informacijske matrike za ta model.

### 4.3 Radij nerazločljivosti

Po standardni asimptotični teoriji nelinearnih najmanjših kvadratov (Seber & Wild, pogl. 5) je porazdelitev cenilke $\hat c$ približno

$$
\hat c \;\approx\; \mathcal N\big(c_0,\; \sigma^2 G^{-1}\big), \tag{4.3}
$$

kjer je $\sigma^2$ raven šuma na residual (reda $\varepsilon^2$ iz poglavja 3). Za lastni vektor $v$ matrike $G$ z lastno vrednostjo $\lambda$ je torej standardna napaka ocene vzdolž $v$ enaka $\sigma/\sqrt\lambda$:

$$
\boxed{\; |\hat c - c_0|_v \;\sim\; \sqrt{\dfrac{\text{raven šuma}}{\lambda}} \;} \tag{4.4}
$$

**Interpretacija.** Smeri s stroje/majhno $\lambda$ (kombinacije parametrov, ki komaj vplivajo na integrirano dinamiko na tem oknu podatkov — npr. produkt dveh hitrostnih konstant, ko je viden le njun produkt) imajo velik radij nerazločljivosti: dva zelo različna $c$ dasta praktično enak $\hat x$ in torej praktično enak loss. Smeri z velikim $\lambda$ so ostro določene.

### 4.4 Strukturna vs. praktična identifikabilnost

Ta analiza spada v širše področje **identifikabilnosti** dinamičnih sistemov:

- *Strukturna identifikabilnost* — vprašanje, ali je $c_0$ enolično določen že iz **popolnih, brezšumnih** opazovanj (lastnost modela, ne podatkov).
- *Praktična identifikabilnost* — ali je $c_0$ enolično določljiv iz **danih, končnih in zašumljenih** podatkov (to je natanko vprašanje iz te sekcije; $G$ je odvisen od mreže $t_i$ in raznolikosti trajektorije).

> **Vir (strukturna):** C. Cobelli, J. J. DiStefano, *Parameter and structural identifiability concepts and ambiguities*, American Journal of Physiology 239(1), 1980, R7–R24.
> **Vir (praktična, profile likelihood):** A. Raue et al., *Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood*, Bioinformatics 25(15), 2009, 1923–1929.

Pomembna posledica konstrukcije: ker $G$ pri integral matchingu nastane z **integriranjem** občutljivosti prek celotnega podintervala (namesto vzorčenja v eni točki, kot pri gradient matchingu — glej poglavje 6), je $G$ tipično bolje pogojena ($\lambda_{\min}$ večji) kot ustrezna matrika pri gradient matchingu, kar pomeni ožje intervale zaupanja za enako количino podatkov.

---

## 5. Vpliv dolžine intervala $\Delta$

### 5.1 Dva režima glede na lengthscale

Groba meja $\|e(t_i)-e(t_{i-1})\|\le 2\varepsilon$ iz (3.2) je realistična le, če sta $t_i,t_{i-1}$ pod GP posteriorjem približno neodvisna, tj. $\Delta_i\gg\ell$ (Definicija 1.2). Ko je $\Delta_i\ll\ell$, je posteriorna povprečna funkcija na tej skali gladka (glej §1.5), zato je (hevristično, iz Höllder/Lipschitzove zveznosti posteriornega povprečja z eksponento, ki jo določa $\ell$):

$$
\|e(t_i)-e(t_{i-1})\| \;\sim\; \varepsilon\cdot\frac{\Delta_i}{\ell} \qquad (\Delta_i\ll\ell). \tag{5.1}
$$

*(Opomba: (5.1) je dimenzijski/hevristični argument, ne strog izrek — natančna oblika je odvisna od izbire jedra; glej vir spodaj za rigorozne različice.)*

Posledica za skupno mejo (3.3), odvisno od režima:

$$
\|l(c_0)\|_1 \sim
\begin{cases}
2n\varepsilon = \dfrac{2\varepsilon T}{\bar\Delta} & \Delta\gg\ell \quad\text{(raste, ko manjšaš } \Delta\text{)}\\[2mm]
\dfrac{\varepsilon T}{\ell} & \Delta\ll\ell \quad\text{(neodvisno od } \Delta\text{)}
\end{cases} \tag{5.2}
$$

> **Vir:** A. W. van der Vaart, J. H. van Zanten (2008), kot v §1.4 — gladkost posteriornega povprečja; Wahba (1990) §1 za RKHS-Höllder ocene.

### 5.2 Grönwallova neenakost in bias pri dolgih intervalih

Drugi učinek dolgega $\Delta_i$: bias-člen $L_x\varepsilon\Delta_i$ v (3.2) je zgolj linearizirana (prvi red) ocena. Za sistem, kjer napaka propagira skozi nelinearen $F$, natančnejšo mejo da **Grönwallova neenakost**:

**Lema (Grönwall, 1919).** Če $u(t)\ge0$ zadošča $u(t)\le\alpha+\beta\int_{t_0}^t u(s)\,ds$ za $\beta\ge0$, potem

$$
u(t) \le \alpha\, e^{\beta(t-t_0)}. \tag{5.3}
$$

Uporabljeno na propagacijo napake znotraj enega (dolgega) intervala pri napačnem $c$: odstopanje trajektorije od prave rešitve lahko raste kot $e^{L_x\Delta_i}$, ne le linearno z $\Delta_i$. To je razlog, da **en sam** zelo dolg interval ($n=1$, t.i. *single shooting*) kaznuje z eksponentno občutljivostjo na $c$ — optimizacijska pokrajina postane skrajno nelinearna in slabo pogojena, posebej pri kaotičnih/togih (stiff) sistemih.

> **Vir:** T. H. Grönwall, *Note on the derivatives with respect to a parameter of the solutions of a system of differential equations*, Annals of Mathematics 20(4), 1919, 292–296.

### 5.3 Single shooting proti multiple shooting

Integral matching z $n>1$ intervali je natanko instanca metode **multiple shooting**: namesto da integriramo od $t_0$ do $t_n$ v enem kosu (kar akumulira Grönwallovo eksponentno napako), *sidramo* trajektorijo nazaj na $\hat x(t_i)$ na vsakem koraku — to prepreči eksponentno kopičenje napake na račun tega, da vpeljemo $n$ ločenih residualov (3.2), ki jih je treba nato uskladiti.

> **Vir:** H. G. Bock, *Recent advances in parameter identification techniques for ODE*, v: P. Deuflhard, E. Hairer (ur.), *Numerical Treatment of Inverse Problems in Differential and Integral Equations*, Birkhäuser, 1983 — izvorni članek o multiple shooting za ODE parameter estimation.

### 5.4 Praktično vodilo

Iz (5.2) + (5.3) + §4.2 (daljši $\Delta_i$ → boljša pogojenost $G$) sledi konkreten kompromis:

- $\Delta \ll \ell$: nič ne pridobiš (formula 5.2, spodnji primer) — le akumuliraš več intervalov brez nove informacije (sosednja $\hat x(t_i),\hat x(t_{i-1})$ sta pod posteriorjem skoraj ista naključna spremenljivka).
- $\Delta \gg \ell$: dno šuma na interval pada s $\Delta$ (5.2, zgornji primer), $G$ je bolje pogojena (§4.2), a Grönwallov bias (5.3) in nelinearnost optimizacije rasteta.

**Priporočilo:** izberi $\Delta_i$ reda velikosti korelacijske dolžine GP jedra $\ell$ (dobiš jo direktno kot hiperparameter iz prileganja GP). To je empirično preverljivo vodilo, ne le teoretična poanta.

---

## 6. Primerjava z gradient matchingom

### 6.1 Definicija gradient matchinga

**Definicija 6.1.** Gradient matching uporabi residual v točkah namesto integriran residual po intervalih:

$$
r_i^{GM}(c) := \hat x'(t_i) - F(\hat x(t_i),t_i,c). \tag{6.1}
$$

> **Vir:** J. O. Ramsay et al. (2007), kot v §3.3; B. Calderhead, M. Girolami, N. D. Lawrence, *Accelerating Bayesian inference over nonlinear differential equations with Gaussian processes*, NeurIPS 21, 2008; P. Wenk et al., *Fast Gaussian Process Based Gradient Matching for Parameter Identification in Systems of Nonlinear ODEs*, AISTATS 2019.

### 6.2 Dobro in slabo postavljeni problemi

**Definicija 6.2 (Hadamard).** Problem je *dobro postavljen (well-posed)*, če ima rešitev, ta je enolična in je **zvezno odvisna od podatkov**. Sicer je *slabo postavljen (ill-posed)*.

Numerično odvajanje zašumljenih podatkov je klasičen primer slabo postavljenega problema: majhna sprememba (šum z amplitudo $\varepsilon$, poljubno visoke frekvence) v vhodni funkciji lahko povzroči poljubno veliko spremembo odvoda. Numerična **integracija** je nasprotno *kompakten* (gladilen) operator — dušilec visokih frekvenc — in je zato dobro postavljena.

> **Vir:** H. W. Engl, M. Hanke, A. Neubauer, *Regularization of Inverse Problems*, Kluwer Academic Publishers, 1996, pogl. 2 — formalna obravnava odvajanja kot ill-posed problema in zakaj integracija ni.

### 6.3 Ojačitev šuma pri odvajanju (hevristična ocena)

Za stacionaren GP z jedrom lengthscale $\ell$ (1.2) posteriorna (ko)varianca odvoda vsebuje $\partial^2 k/\partial t\partial t'$, ki je za (1.2) enaka $\sigma_f^2/\ell^2$ pri $t=t'$ — dimenzijsko to pomeni, da je negotovost odvoda za faktor $\sim 1/\ell$ večja od negotovosti same funkcije:

$$
\|e'(t)\| \;\sim\; \frac{\varepsilon}{\ell}. \tag{6.2}
$$

Torej ima gradient-matching residual (6.1) dno šuma reda $\varepsilon/\ell$ **na točko**, medtem ko ima integral-matching residual (2.4) dno šuma reda $\varepsilon$ **na (dovolj dolg) interval** — brez faktorja $1/\ell$.

> **Vir:** Rasmussen & Williams (2006), §9.4 (eksplicitna formula za kovarianco GP odvodov).

### 6.4 Limita $\Delta\to0$: gradient matching kot degeneriran primer

Po izreku o povprečni vrednosti za integral, ko $\Delta_i\to0$:

$$
l_i(c) = \int_{t_{i-1}}^{t_i}\!\big[\hat x'(\tau)-F(\hat x(\tau),\tau,c)\big]d\tau + O(\Delta_i^2) \;\approx\; \Delta_i\cdot r_i^{GM}(c). \tag{6.3}
$$

Torej integral matching **zvezno degenerira** v gradient matching, ko $\Delta\to0$; iz (5.2) in (6.2) se dni ujemata natanko pri $\Delta\sim\ell$, kar potrjuje vodilo iz §5.4 iz drugega zornega kota.

### 6.5 Primerjalna tabela

| Lastnost | Gradient matching | Integral matching ($\Delta\gtrsim\ell$) |
|---|---|---|
| Operacija na $\hat x$ | odvajanje (ill-posed) | integracija (well-posed, gladi) |
| Dno šuma na enoto podatkov | $O(\varepsilon/\ell)$ | $O(\varepsilon)$, neodvisno od $\ell$ |
| Pogojenost $G$ (identifikabilnost) | točkovna občutljivost | integrirana občutljivost — praviloma bolje pogojena |
| Bias pri nelinearnem $F$ | majhen (lokalna ocena) | raste z $\Delta$ (Grönwall, §5.2) |
| Degenerira v | — | gradient matching, ko $\Delta\to0$ |

**Odgovor na vprašanje "ali je integral matching bolj odporen na šum":** da, formalno, v smislu §6.2–6.3 — a le pri $\Delta\gtrsim\ell$. Pri $\Delta\to0$ prednost izgine, saj metodi postaneta ekvivalentni.

---

## 7. Povzetek in praktična priporočila

1. **Kriterij ustavitve** (§3): ne uporabljaj analitične meje (3.3) neposredno (odvisna je od neznanega $L_x$); kalibriraj prag za $J(c)$ z bootstrapom iz GP posteriorja.
2. **Identifikabilnost** (§4): izračunaj $G=\sum_i S_i^\top S_i$ in njegov spekter; majhne lastne vrednosti razkrijejo kombinacije parametrov, ki jih ti podatki ne morejo ločiti.
3. **Dolžina intervala** (§5): izberi $\Delta$ reda velikosti GP-jeve lengthscale $\ell$; ne sekaj bolj fino (brez pridobitve, akumuliraš šum) niti bistveno bolj grobo (Grönwallov bias, single-shooting nestabilnost).
4. **Primerjava z gradient matchingom** (§6): integral matching je matematično utemeljeno bolj odporen na šum, ker integracija deluje kot dobro postavljen (gladilni) operator v primerjavi z odvajanjem — a ta prednost je pogojena s $\Delta\gtrsim\ell$, sicer degenerira nazaj v gradient matching.

---

## Literatura

1. C. E. Rasmussen, C. K. I. Williams. *Gaussian Processes for Machine Learning*. MIT Press, 2006.
2. A. W. van der Vaart, J. H. van Zanten. "Rates of contraction of posterior distributions based on Gaussian process priors." *Annals of Statistics* 36(3), 2008, 1435–1463.
3. G. Wahba. *Spline Models for Observational Data*. SIAM, 1990.
4. W. Rudin. *Principles of Mathematical Analysis*, 3rd ed. McGraw-Hill, 1976.
5. E. A. Coddington, N. Levinson. *Theory of Ordinary Differential Equations*. McGraw-Hill, 1955.
6. S. Boucheron, G. Lugosi, P. Massart. *Concentration Inequalities: A Nonasymptotic Theory of Independence*. Oxford University Press, 2013.
7. J. O. Ramsay, G. Hooker, D. Campbell, J. Cao. "Parameter estimation for differential equations: a generalized smoothing approach." *JRSS-B* 69(5), 2007, 741–796.
8. G. A. F. Seber, C. J. Wild. *Nonlinear Regression*. Wiley, 2003.
9. C. Cobelli, J. J. DiStefano. "Parameter and structural identifiability concepts and ambiguities." *American Journal of Physiology* 239(1), 1980, R7–R24.
10. A. Raue et al. "Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood." *Bioinformatics* 25(15), 2009, 1923–1929.
11. T. H. Grönwall. "Note on the derivatives with respect to a parameter of the solutions of a system of differential equations." *Annals of Mathematics* 20(4), 1919, 292–296.
12. H. G. Bock. "Recent advances in parameter identification techniques for ODE." In *Numerical Treatment of Inverse Problems in Differential and Integral Equations*, eds. P. Deuflhard, E. Hairer. Birkhäuser, 1983.
13. B. Calderhead, M. Girolami, N. D. Lawrence. "Accelerating Bayesian inference over nonlinear differential equations with Gaussian processes." *NeurIPS* 21, 2008.
14. P. Wenk, A. Gotovos, S. Bauer, N. S. Gorbach, A. Krause, J. M. Buhmann. "Fast Gaussian Process Based Gradient Matching for Parameter Identification in Systems of Nonlinear ODEs." *AISTATS*, 2019.
15. H. W. Engl, M. Hanke, A. Neubauer. *Regularization of Inverse Problems*. Kluwer Academic Publishers, 1996.
16. J. Hadamard. "Sur les problèmes aux dérivées partielles et leur signification physique." *Princeton University Bulletin* 13, 1902, 49–52.
