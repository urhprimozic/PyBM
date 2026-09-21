# Gradient matching: matematične osnove

*Analiza napake, identifikabilnosti in občutljivosti na šum pri ocenjevanju parametrov ODE iz
točkovnega (ne integriranega) ujemanja z zašumljenimi podatki. Spremljevalni dokument k
`integral-matching.md` - iste definicije, ista notacija, kjer se da neposredno primerljivo.*

---

## Kako brati ta učbenik

Ista konvencija kot `integral-matching.md`: vsako poglavje uvede pojem, ga formalno definira, poda
vir, nato ga takoj uporabi v izpeljavi. Kjer trditev ni strogo dokazana, ampak je hevristični
(dimenzijski) argument, je to eksplicitno označeno. Kjer se pojem natančno ujema z definicijo v
`integral-matching.md`, je ponovljen na kratko (za samostojnost dokumenta), s sklicem nazaj za polno
izpeljavo.

Notacija je ista kot v `integral-matching.md`:

- $x(t)\in\mathbb R^d$ - prava (neznana) rešitev ODE,
- $c\in\mathbb R^p$ - vektor parametrov, $c_0$ njegova prava vrednost,
- $F:\mathbb R^d\times\mathbb R\times\mathbb R^p\to\mathbb R^d$ - desna stran enačbe, $x'=F(x,t,c)$,
- $y_i = x(t_i) + \eta_i$, $\eta_i\sim$ šum - opazovani podatki,
- $\hat x(t)$ - GP posteriorna povprečna funkcija, ocenjena iz $\{(t_i,y_i)\}$,
- $t_0<t_1<\dots<t_n$ - mreža časov.

Novo v tem dokumentu:

- $\hat x'(t)$ - GP posteriorna povprečna funkcija ODVODA (definirana spodaj, §1.2),
- $r_i(c) := \hat x'(t_i) - F(\hat x(t_i),t_i,c)$ - gradient-matching residual v točki $t_i$,
- $\ell$ - korelacijska dolžina (lengthscale) GP-jevega jedra (glej `integral-matching.md` §1.2),
- $\varepsilon$ - enakomerna meja za $\|\hat x(t)-x(t)\|$ (glej §1.4 spodaj, enako kot prej),
- $\varepsilon'$ - enakomerna meja za $\|\hat x'(t)-x'(t)\|$ (NOVO - glavni predmet tega dokumenta).

---

## 1. Gradniki

### 1.1 Problem ocenjevanja parametrov ODE

Enak inverzni problem kot v `integral-matching.md` §1.1: dana družina ODE $x'=F(x,t,c)$ in
zašumljena opazovanja $y_i=x(t_i)+\eta_i$; cilj je oceniti $c_0$. Isti Hadamardov okvir
(dobro/slabo postavljeni problemi) se uporabi spodaj v §6.2 - tokrat na operaciji ODVAJANJA, ne
integriranja.

### 1.2 Gaussovski proces IN NJEGOV ODVOD

**Definicija 1.2** (ponovitev, glej `integral-matching.md` §1.2). $\hat x$ je GP z jedrom $k(t,t')$;
posteriorna povprečna funkcija je $\hat x(t) = k(t)^\top(K+\sigma^2 I)^{-1}y$.

**Definicija 1.2' (odvod GP-ja).** Če je $f\sim\mathcal{GP}(\mu,k)$ in je $k$ dvakrat odvedljiv, je
$f'$ prav tako Gaussovski proces (GP je zaprt pod linearnimi operatorji - odvajanje je linearno), s

$$
\operatorname{Cov}[f'(t),f'(t')] = \frac{\partial^2 k(t,t')}{\partial t\,\partial t'}, \qquad
\operatorname{Cov}[f'(t),f(t')] = \frac{\partial k(t,t')}{\partial t}. \tag{1.1}
$$

Posteriorna povprečna funkcija in varianca ODVODA (pogojeno na $y_i=f(t_i)+\eta_i$) sta

$$
\hat x'(t^*) = k'_*(t^*)^\top(K+\sigma^2I)^{-1}y, \qquad
\operatorname{Var}_{\text{post}}[\hat x'(t^*)] = k''(t^*,t^*) - k'_*(t^*)^\top(K+\sigma^2I)^{-1}k'_*(t^*),
\tag{1.2}
$$

kjer je $k'_*(t)_i := \partial k(t,t_i)/\partial t$. To je natanko `_kernel_deriv_wrt_first`
(posteriorno povprečje) in `_FittedGP.var_deriv` (posteriorna varianca) v
`pybm/estimate/gradient_matching.py` - obe formuli v kodi realizirani prek shranjenega
Choleskyjevega faktorja $L$ ($K+\sigma^2I=LL^\top$), z $v(t):=L^{-1}k(t)$, tako da je
$k(t)^\top(K+\sigma^2I)^{-1}k(t')=v(t)^\top v(t')$ - ista konstrukcija kot pri `_FittedGP.var`.

> **Vir:** E. Solak, R. Murray-Smith, W. E. Leithead, D. J. Leith, C. E. Rasmussen, *Derivative
> observations in Gaussian process models of dynamic systems*, NeurIPS 16, 2003 - izvorna izpeljava
> (1.1)-(1.2) za GP-je z odvodnimi opazovanji/poizvedbami. C. E. Rasmussen, C. K. I. Williams,
> *Gaussian Processes for Machine Learning*, MIT Press, 2006, §9.4 - ista formula, standardni
> učbeniški zapis.

### 1.3 Lipschitzova zveznost

Enaka Definicija 1.3 kot v `integral-matching.md` §1.3: $F$ je $L_x$-Lipschitzova v $x$ in
$L_c$-Lipschitzova v $c$. Spet edino, kar potrebujemo o $F$ samem.

### 1.4 Enakomerna napaka interpolacije IN NJENEGA ODVODA

**Definicija 1.4** (ponovitev). $e(t):=\hat x(t)-x(t)$, $\sup_t\|e(t)\|\le\varepsilon$ z visoko
verjetnostjo (glej `integral-matching.md` §1.4 za vir - posteriorna kontrakcija GP regresije).

**Definicija 1.4' (NOVO).** Naj bo $e'(t):=\hat x'(t)-x'(t)$ napaka ODVODNE ocene ($x'$ je odvod
prave, deterministične rešitve - ne slučajna spremenljivka; ves šum je v $\hat x'$). Predpostavimo
$\varepsilon'>0$, tako da z visoko verjetnostjo

$$
\sup_t\|e'(t)\|\le\varepsilon'. \tag{1.3}
$$

$\varepsilon'$ je - tako kot $\varepsilon$ - eksogen v tem dokumentu; §1.5 spodaj izpelje natančno
razmerje med $\varepsilon'$ in $\varepsilon$, ki je JEDRO tega celotnega dokumenta.

### 1.5 Razmerje $\varepsilon'$ proti $\varepsilon$: zakaj odvajanje ojači šum

To je natančna, ne le hevristična izpeljava trditve, ki je bila v `integral-matching.md` §6.3
podana kot dimenzijski argument.

**Trditev 1.5 (apriorno razmerje za stacionarno jedro).** Za stacionarno jedro
$k(t,t')=\sigma_f^2\rho((t-t')/\ell)$ z gladkim, sodim $\rho$, $\rho(0)=1$, je apriorna varianca
funkcije $\operatorname{Var}[f(t)]=k(t,t)=\sigma_f^2$, apriorna varianca odvoda pa (iz (1.1) pri
$t=t'$, uporabljeno na kvadratno eksponentno jedro (1.2) iz `integral-matching.md`)

$$
\operatorname{Var}[f'(t)] = \left.\frac{\partial^2k(t,t')}{\partial t\,\partial t'}\right|_{t=t'}
= \frac{\sigma_f^2}{\ell^2}. \tag{1.4}
$$

*Izpeljava za RBF jedro $k(t,t')=\sigma_f^2\exp(-(t-t')^2/(2\ell^2))$:*

$$
\frac{\partial k}{\partial t} = -\sigma_f^2\,\frac{t-t'}{\ell^2}\exp\!\left(-\frac{(t-t')^2}{2\ell^2}\right), \qquad
\frac{\partial^2k}{\partial t\,\partial t'} = \frac{\sigma_f^2}{\ell^2}\exp\!\left(-\frac{(t-t')^2}{2\ell^2}\right)\left[1-\frac{(t-t')^2}{\ell^2}\right].
$$

Pri $t=t'$ oklepaj postane $1$, kar da natanko (1.4). $\blacksquare$

**Posledica (razmerje standardnih deviacij).**

$$
\boxed{\;\operatorname{std}[f'(t)] = \frac{\operatorname{std}[f(t)]}{\ell}\;} \tag{1.5}
$$

**Od apriorne k posteriorni trditvi (previdnost).** (1.4)-(1.5) sta EKSAKTNI za apriorno (brez
podatkov) porazdelitev. Za POSTERIORNO varianco (1.2), ki jo dejansko uporabljamo (`.var`,
`.var_deriv` v kodi), enako razmerje ni algebraično eksaktno - oba, $\operatorname{Var}_{\text{post}}[\hat x(t)]$
in $\operatorname{Var}_{\text{post}}[\hat x'(t)]$, se zmanjšata glede na podatke, a NE nujno za isti
faktor. Empirično (glej `pybm.benchamark.benchmark_2`, GP na Bled `phyto.conc`) razmerje ostane
blizu $1/\ell$, kadar je gostota podatkov glede na $\ell$ približno enakomerna - to je razlog, da v
`integral-matching.md` §6.3 ta trditev nastopi kot "dimenzijska ocena", ne strog izrek. V tem
dokumentu je (1.4)-(1.5) POTRJENA kot eksakten rezultat na apriorni ravni - kar je dovolj močna
osnova za vso nadaljnjo analizo, ker gradient matching implicitno primerja $\varepsilon'$ (odvod)
in $\varepsilon$ (vrednost) na ISTI podatkovni gostoti, kjer aproksimacija $\varepsilon'\approx\varepsilon/\ell$
najbolje drži.

**Zakaj to pomeni "bolj občutljiv na šum".** $\varepsilon$ je raven šuma v $\hat x$ sami; $\varepsilon'=\varepsilon/\ell$
je raven šuma v $\hat x'$. Ker je $\ell$ tipično MANJŠI od $1$ v enotah, v katerih so podatki
naravno merjeni (lengthscale je "kako hitro se stvari spreminjajo", ne absolutna skala), je
$1/\ell$ tipično FAKTOR OJAČITVE, ne dušenja. Gradient matching primerja $F$ neposredno s to ojačano
količino ($\hat x'$) - integral matching pa NIKOLI ne rabi $\hat x'$ sploh (glej §6 spodaj).

---

## 2. Metoda gradient matching

### 2.1 Definicija izgube

**Definicija 2.1.** Za dano množico kolokacijskih točk $t_1,\dots,t_m$ (lahko, a ni nujno, enaka
mreži podatkov - v kodi `collocation_times`, privzeto `t_eval`) in kandidatni parameter $c$
definiramo

$$
r_i(c) := \hat x'(t_i) - F(\hat x(t_i),t_i,c), \qquad i=1,\dots,m, \tag{2.1}
$$

in $J(c)=\sum_i\|r_i(c)\|^2$. Ocena je $\hat c=\arg\min_c J(c)$.

V primerjavi z integral matchingovim (2.1) (glej `integral-matching.md`) tu NI integrala: vsaka
$r_i$ je TOČKOVNA, neodvisna preveritev, ali $\hat x$ lokalno reši ODE v odvodni (močni) obliki.

### 2.2 Razčlenitev $r_i(c)$

Ker $x$ reši ODE pri $c_0$: $x'(t)=F(x(t),t,c_0)$. Ker je $x$ deterministična funkcija (ves šum je
v $\hat x$), je $e'(t)=\hat x'(t)-x'(t)$ dobro definiran (Definicija 1.4'), torej

$$
\hat x'(t_i) = x'(t_i) + e'(t_i) = F(x(t_i),t_i,c_0) + e'(t_i). \tag{2.2}
$$

Vstavimo v (2.1):

$$
r_i(c) = e'(t_i) + \big[F(x(t_i),t_i,c_0) - F(\hat x(t_i),t_i,c)\big]. \tag{2.3}
$$

Znotraj oglatega oklepaja prištejemo in odštejemo $F(\hat x(t_i),t_i,c_0)$ (enaka Lipschitzova
razčlenitev kot `integral-matching.md` (3.1)):

$$
F(x,t_i,c_0)-F(\hat x,t_i,c) = \underbrace{\big[F(x,t_i,c_0)-F(\hat x,t_i,c_0)\big]}_{\le L_x\|e(t_i)\|}
+ \underbrace{\big[F(\hat x,t_i,c_0)-F(\hat x,t_i,c)\big]}_{\text{signal o } c-c_0}. \tag{2.4}
$$

$$
\boxed{\; r_i(c) = \underbrace{e'(t_i)}_{\text{odvodni šum, red } \varepsilon/\ell}
\;+\; \underbrace{\big[F(x(t_i),t_i,c_0)-F(\hat x(t_i),t_i,c_0)\big]}_{\text{vrednostni šum, red } L_x\varepsilon}
\;+\; \underbrace{\big[F(\hat x(t_i),t_i,c_0)-F(\hat x(t_i),t_i,c)\big]}_{\text{signal o } c} \;} \tag{2.5}
$$

**Primerjava z integral matchingovim (2.4).** Strukturno enaka razčlenitev (šum + šum + signal), a
z eno ključno razliko: integral matchingov "šumski" člen je $e(t_i)-e(t_{i-1})$ - razlika DVEH
vrednostnih napak, ki lahko z Delta-jem (širino okna) postane poljubno majhna relativno na signal
(glej `integral-matching.md` §5.1). Gradient matchingov prvi šumski člen $e'(t_i)$ je fiksiran na
odvodno skalo $\varepsilon/\ell$ - NI načina, da bi ga s katerokoli izbiro kolokacijskih točk
zmanjšali (edino zmanjšanje $\varepsilon$ same, tj. boljši/gostejši podatki, pomaga - glej §4.4
spodaj). To je natančna matematična oblika "gradient matching je bolj občutljiv na šum".

---

## 3. Meja za izgubo pri $c_0$ in kriterij ustavitve

### 3.1 Groba (worst-case) meja

Z (1.3), (1.4)-Definicijo 1.4 in trikotniško neenakostjo iz (2.5) pri $c=c_0$ (zadnji člen odpade):

$$
\|r_i(c_0)\| \le \varepsilon' + L_x\varepsilon. \tag{3.1}
$$

Seštejemo po $i=1,\dots,m$:

$$
\boxed{\; \|r(c_0)\|_1 = \sum_{i=1}^m\|r_i(c_0)\| \;\le\; m(\varepsilon'+L_x\varepsilon) \;} \tag{3.2}
$$

**Primerjava z integral matchingovo (3.3)** ($\|l(c_0)\|_1\le 2n\varepsilon+L_x\varepsilon T$): tam
je meja odvisna od ŠTEVILA OKEN $n$ in RAZPONA $T$ (prek Lipschitzovega člena); tu je odvisna od
ŠTEVILA KOLOKACIJSKIH TOČK $m$ in - prek $\varepsilon'=\varepsilon/\ell$ - od GP-jeve lengthscale
same, ne od razpona podatkov. Uporabimo (1.5):

$$
\|r(c_0)\|_1 \;\lesssim\; m\varepsilon\left(\frac1\ell + L_x\right). \tag{3.3}
$$

**Kdaj prevlada odvodni šum.** Člen $1/\ell$ prevlada nad $L_x$, kadar je $\ell < 1/L_x$ - tj. kadar
GP-jeva korelacijska dolžina ni veliko večja od karakteristične časovne skale dinamike same
($1/L_x$ ima enote časa, ker je $L_x$ definiran prek $\|F(x)-F(x')\|\le L_x\|x-x'\|$, torej groba
"časovna konstanta" sistema). To je tipičen režim na realnih, ne pretirano gladko vzorčenih
podatkih (glej §7) - v tem režimu je (3.3) DOMINIRANA z odvodnim šumom, ne z modelovo lastno
občutljivostjo $L_x$: **nizka $J(c_0)$ meja gradient matchinga je v praksi skoraj v celoti
posledica $\varepsilon/\ell$, ne prave informacije o dinamiki**.

### 3.2 Zakaj je (3.2) preohlapna

Enak argument kot `integral-matching.md` §3.2: vsota $m$ absolutnih vrednosti prek trikotniške
neenakosti je pesimistična, če so $e'(t_i)$ približno neodvisne (kar velja za $t_i$ narazen za
$\gg\ell$ - glej §4.4 spodaj) slučajne spremenljivke s povprečjem nič. Koncentracijske neenakosti
(Bernstein) dajo za $J(c_0)=\sum_i\|r_i(c_0)\|^2$ mejo reda $O(m\varepsilon'^2)=O(m\varepsilon^2/\ell^2)$
z nihanji $O(\sqrt m\,\varepsilon^2/\ell^2)$ - ostrejše od naivne $O(m^2\varepsilon^2/\ell^2)$.

> **Vir:** enak kot `integral-matching.md` §3.2 - S. Boucheron, G. Lugosi, P. Massart,
> *Concentration Inequalities*, Oxford University Press, 2013.

### 3.3 Praktičen kriterij ustavitve

Enak parametrični-bootstrap pristop kot `integral-matching.md` §3.3, tu neposredno prenosljiv: iz
GP posteriorja (za $\hat x'$, ne $\hat x$) vzorčimo realizacije, prerašunamo $J$ pri trenutnem
$\hat c$, dobimo empirično ničelno porazdelitev $J(c_0)$. Ni (še) implementirano v
`pybm.estimate.gradient_matching` (glej §7, praktična priporočila).

> **Vir:** J. O. Ramsay, G. Hooker, D. Campbell, J. Cao, *Parameter estimation for differential
> equations: a generalized smoothing approach*, JRSS-B 69(5), 2007, 741-796.

---

## 4. Identifikabilnost

### 4.1-4.2 Fisherjeva informacija in Gauss-Newton linearizacija

Enaka konstrukcija kot `integral-matching.md` §4.1-4.2, s pointwise (ne integrirano) matriko
občutljivosti:

$$
S_i^{GM} := \frac{\partial F}{\partial c}\big(\hat x(t_i),t_i,c_0\big) \in\mathbb R^{d\times p},
\qquad G^{GM} := \sum_{i=1}^m (S_i^{GM})^\top S_i^{GM}. \tag{4.1}
$$

Primerjaj z integral matchingovim $S_i^{IM}=\int_{t_{i-1}}^{t_i}\partial F/\partial c\,d\tau$
(`integral-matching.md` (4.1)) - INTEGRIRANA, ne pointwise, občutljivost.

Ista asimptotska teorija (Seber & Wild, pogl. 5) da $\hat c\approx\mathcal N(c_0,\sigma^2(G^{GM})^{-1})$,
isti radij nerazločljivosti $|\hat c-c_0|_v\sim\sqrt{\text{raven šuma}/\lambda}$ za lastni vektor
$v$ matrike $G^{GM}$ z lastno vrednostjo $\lambda$ (`integral-matching.md` Eq. 4.4).

`pybm.estimate.gradient_matching._identifiability_report` izračuna to natanko - iz
`least_squares_result.jac` (že izračunan Jacobijan pri konvergenci) prek SVD, $G^{GM}=J_r^\top J_r$
z lastnimi vrednostmi $=$ kvadrati singularnih vrednosti $J_r$. Enaka implementacija se uporablja
za `IntegralMatchingResult.identifiability_report()` - ista funkcija, različen $J_r$ na vhodu.

### 4.3 Strukturna vs. praktična identifikabilnost

Enaka razlika kot `integral-matching.md` §4.4 - strukturna (iz brezšumnih, popolnih podatkov) proti
praktični (iz danih, končnih, zašumljenih podatkov, torej odvisna od $G$).

### 4.4 Zakaj je $G^{GM}$ tipično slabše pogojena kot $G^{IM}$ (hevristični argument)

*(Ta trditev je bila v `integral-matching.md` §4.4 zapisana kot posledica brez izpeljave -
tukaj podana natančneje, a še vedno kot kvalitativen, ne strog argument.)*

Integracija je LINEARNA GLAJENJA/POVPREČENJA operacija: $S_i^{IM}$ je (do faktorja širine okna)
POVPREČJE $\partial F/\partial c$ prek mnogih bližnjih $\tau$ znotraj okna, medtem ko je $S_i^{GM}$
ovrednoten v ENI SAMI točki $t_i$. Povprečenje teži k temu, da "zapolni" smeri v parametrskem
prostoru, ki so ob KATERIKOLI POSAMEZNI točki šibko vzbujene, a se kumulativno prek okna
seštejejo v znaten prispevek - podoben pojav kot pri numeričnem računanju ranga: vsota (integral)
matrik ima praviloma manj/manjše ničelne singularne vrednosti kot posamezna matrika sama.
To NI dokazana neenakost za splošen $F$, je pa skladno z empiričnim opažanjem na Bled podatkih
(`pybm.benchamark.benchmark_2`): $G^{GM}$-jev `identifiability_report()` je v praksi našel več in
bolj degenerirane (`relative_scale`) smeri kot ustrezen $G^{IM}$ na isti strukturi.

### 4.5 Gostota kolokacijskih točk relativno na $\ell$

Direktna posledica GP-jeve korelacijske strukture (Definicija 1.2): za $|t_i-t_j|\ll\ell$ sta
$\hat x(t_i)$ in $\hat x(t_j)$ (in posledično implicirana $S_i^{GM}$, $S_j^{GM}$) skoraj isti
naključni spremenljivki pod GP posteriorjem - dodajanje kolokacijskih točk gosteje kot $\ell$ NE
doda novih neodvisnih omejitev v $G^{GM}$, enako kot `integral-matching.md` §5.4 ugotovi za okna
ožja od $\ell$ pri integral matchingu. To je gradient matchingov analog tistega vodila: **razmik
kolokacijskih točk naj bo reda $\ell$**, ne gostejši - gostejša mreža poveča $m$ (in navidezno
$G^{GM}$), a brez prave dodatne informacije, poveča pa računski strošek in tveganje numerične
skoraj-singularnosti $G^{GM}$ (skoraj-podvojeni stolpci/vrstice).

`pybm.estimate.integral_matching._auto_stride` to že implementira za integral matching (izbira
`stride` iz GP-jeve lengthscale); enakovredna "auto" izbira razmika kolokacijskih točk za gradient
matching (danes `collocation_times` privzeto kar `t_eval`, brez posebne obravnave glede na $\ell$)
še ni implementirana - glej §7.

---

## 5. Overfitting pri redkih, lokalnih preveritvah

To je bil izvoren motiv za razvoj integral matchinga (glej `pybm.estimate.integral_matching`-jev
modulski docstring, točka 2), tu formaliziran.

**Trditev (kvalitativna).** Gradient matching preveri, ali $\hat x$ lokalno reši ODE, na $m$
IZOLIRANIH točkah. Če ima model $p$ prostih parametrov s $p$ primerljiv ali večji od ŠTEVILA
UČINKOVITO NEODVISNIH omejitev $m_{\text{eff}}\sim(\text{razpon podatkov})/\ell$ (ne surovega $m$ -
glej §4.5), je sistem $\{r_i(c)=0\}_{i=1}^m$ podpoločen v smislu, da obstaja $c\ne c_0$ z majhnim
(ali celo $\approx 0$) $J(c)$, ki NE reproducira prave trajektorije MED kolokacijskimi točkami -
$F$ z dovolj svobode se lahko "prilagodi" vsaki izolirani zahtevi posebej, brez globalne
konsistentnosti, ki bi jo prava integracija/simulacija zahtevala.

To je NATANKO opažanje na realnem Bled modelu (glej `pybm.benchamark.benchmark_2`-jev modulski
docstring): majhen gradient-matching rezidual, a naprej-simulirana trajektorija, ki močno odstopa
od podatkov. Formalno je to isti pojav kot majhna lastna vrednost v $G^{GM}$ (§4) - "podpoločenost"
in "overfitting" sta tu dva jezika za isto stvar: obstaja smer v prostoru parametrov, vzdolž katere
$J(c)$ skoraj ne raste, torej optimizator lahko po tej smeri poljubno "zdrsne" stran od $c_0$ brez
kazni v svoji lastni (točkovni) izgubi.

**Zakaj integral matching to omili (ne odpravi).** Integracija prek okna zahteva, da $F$ pri danem
$c$ pravilno napove AKUMULIRANO spremembo prek celotnega okna, kar je strožja, manj lokalna zahteva
kot ujemanje v eni sami točki - manj "prostora" za $F$, da se prilagodi izolirani zahtevi brez
globalne konsistentnosti. To NE odpravi podpoločenosti nasploh (glej `integral-matching.md` §5.2 -
Grönwallova pristranskost pri zelo širokih oknih uvede svoj lasten problem), je pa strožja preveritev
za PRIMERLJIVO število prostih parametrov.

---

## 6. Primerjava z integral matchingom

### 6.1 Viri za gradient matching kot metodo

> **Vir:** J. O. Ramsay et al. (2007), kot v §3.3; B. Calderhead, M. Girolami, N. D. Lawrence,
> *Accelerating Bayesian inference over nonlinear differential equations with Gaussian processes*,
> NeurIPS 21, 2008 - GP-based gradient matching kot Bayesovska metoda; P. Wenk, A. Gotovos,
> S. Bauer, N. S. Gorbach, A. Krause, J. M. Buhmann, *Fast Gaussian Process Based Gradient Matching
> for Parameter Identification in Systems of Nonlinear ODEs*, AISTATS 2019 - hitra (ne-Bayesovska,
> point-estimate) različica, najbližja implementaciji v `pybm.estimate.gradient_matching`.

### 6.2 Dobro in slabo postavljeni problemi (ponovitev)

Enaka Definicija 6.2 (Hadamard) kot `integral-matching.md` §6.2. Numerično ODVAJANJE zašumljenih
podatkov je slabo postavljen problem (majhen šum $\to$ poljubno velika sprememba odvoda);
INTEGRACIJA je nasprotno kompakten (gladilen) operator, dobro postavljen. Gradient matching stoji
na napačni (slabo postavljeni) strani te delitve - integral matching na pravilni.

> **Vir:** H. W. Engl, M. Hanke, A. Neubauer, *Regularization of Inverse Problems*, Kluwer, 1996,
> pogl. 2.

### 6.3 Limita $\ell\to0$ oz. $\Delta\to0$: ista stvar z dveh strani

`integral-matching.md` §6.4 pokaže, da integral matching zvezno degenerira v gradient matching, ko
$\Delta\to0$ (Taylor/MVT: $l_i(c)\approx\Delta_i\cdot r_i^{GM}(c)$). Ta dokument doda drugo polovico
slike: gradient matching SAM nima analogne "širine" parametra, ki bi ga lahko poljubno povečeval -
edini prosti "gumb" na strani gradient matchinga je GOSTOTA/RAZPORED kolokacijskih točk ($m$, §4.5),
ne kaka $\Delta$. To je STRUKTURNA asimetrija med metodama, ne le tehnična: integral matching ima en
dodaten prost parameter ($\Delta$/`stride`), ki gradient matchingu manjka - in prav ta parameter je
tisti, ki v `integral-matching.md` §5 omogoči nadzor nad kompromisom šum/pristranskost. Gradient
matching tega kompromisa preprosto nima na voljo; njegov edini vzvod (§4.5, gostota točk) vpliva na
$G$ (identifikabilnost), NE na $\varepsilon'$ samo (šumni pod, glej §1.5 - $\varepsilon'$ je
odvisen od $\ell$ in kvalitete podatkov, ne od tega, KOLIKO kolokacijskih točk postavimo).

### 6.4 Primerjalna tabela

| Lastnost | Gradient matching | Integral matching ($\Delta\gtrsim\ell$) |
|---|---|---|
| Operacija na $\hat x$ | odvajanje (ill-posed) | integracija (well-posed, gladi) |
| Šumni pod na enoto | $O(\varepsilon/\ell)$ na TOČKO | $O(\varepsilon)$ na OKNO, neodvisno od $\ell$ |
| Prost parameter za kompromis šum/pristranskost | noben (§6.3) | $\Delta$ (`stride`) |
| Občutljivostna matrika $S_i$ | pointwise, $\partial F/\partial c$ v $t_i$ | integrirana, $\int\partial F/\partial c\,d\tau$ |
| Tipična pogojenost $G$ | slabša (§4.4, hevristika) | boljša (§4.4, hevristika) |
| Tveganje | overfitting na izolirane točke (§5) | Grönwallova pristranskost pri širokih oknih |
| Degenerira v | integral matching, ko $\Delta\to0$ | gradient matching, ko $\Delta\to0$ |

**Odgovor na "zakaj smo bolj občutljivi na šum kot pri integriranju":** formalno, ker gradient
matching primerja $F$ neposredno z $\hat x'$, katerega lastni šum $\varepsilon'\approx\varepsilon/\ell$
JE STRUKTURNO OJAČEN glede na $\varepsilon$ (Trditev 1.5, eksaktna na apriorni ravni) - integral
matching pa NIKOLI ne rabi $\hat x'$, njegov levi del residuala ($\hat x(t_{i+1})-\hat x(t_i)$) je
eksakten prek fundamentalnega izreka in nosi šum reda $\varepsilon$ SAM (ne $\varepsilon/\ell$), dodatno
zmanjšljiv s širino okna (`integral-matching.md` Eq. 5.2). To ni le empirično opažanje - je
neposredna posledica tega, da je odvajanje slabo, integracija pa dobro postavljena operacija na
zašumljenih podatkih (§6.2).

---

## 7. Povzetek in praktična priporočila

1. **Šumni pod** (§1.5, §3): $\varepsilon'\approx\varepsilon/\ell$ je STRUKTURNA, ne odpravljiva
   lastnost gradient matchinga - noben izbor kolokacijskih točk je ne zmanjša. Edino boljši podatki
   (manjši $\varepsilon$, torej gostejši/manj zašumljeni observations, ki jih GP fita) pomagajo.
2. **Identifikabilnost** (§4): izračunaj $G^{GM}=\sum_i(S_i^{GM})^\top S_i^{GM}$ (že
   implementirano: `GradientMatchingResult.identifiability_report()`); pričakuj slabšo pogojenost
   kot pri integral matchingu na isti strukturi (§4.4, hevristika - preveri direktno na svojih
   podatkih, ne predpostavljaj).
3. **Gostota kolokacijskih točk** (§4.5): reda $\ell$, ne gostejše - gostejša mreža ne doda prave
   informacije, poveča pa računski strošek in numerično skoraj-singularnost $G^{GM}$. `_auto_stride`
   (integral matching) obstaja; enakovredna avtomatska izbira za gradient matching NI (še)
   implementirana - naraven kandidat za prihodnje delo, glej `_FittedGP.lengthscale` (že
   izračunan, samo ni uporabljen za to na strani gradient matchinga).
4. **Overfitting na izolirane točke** (§5): majhen $J(\hat c)$ NE zagotavlja, da $\hat c\approx c_0$
   ali da fitani model reproducira pravo trajektorijo MED kolokacijskimi točkami - preveri z resnično
   naprej-simulirano trajektorijo (`_singleshooting_loss`/`_test_mse`), ne zaupaj $J(\hat c)$ sami.
5. **Uteževanje po GP-jevi negotovosti** (`weight_by_uncertainty=True`, glej `_FittedGP.var_deriv`):
   implementirano in matematično korektno (uteži nižje kolokacijske točke, kjer je $\operatorname{Var}[\hat x'(t)]$
   velika), a EMPIRIČNO na Bled podatkih dalo mešane rezultate (enkrat bolje, enkrat slabše -
   preverjeno direktno, glej chat log 2026-08-26) - privzeto izklopljeno, ni dokazana izboljšava.
6. **Praktičen kriterij ustavitve** (§3.3): bootstrap iz GP posteriorja - opisan v teoriji, NI (še)
   implementiran v `pybm.estimate.gradient_matching`.

---

## Literatura

Deljena z `integral-matching.md` (isti viri za GP regresijo, RKHS, Lipschitz, Grönwall,
identifikabilnost, Hadamard, regularizacijo inverznih problemov - glej tam za polni seznam). Novi
viri, specifični za ta dokument:

1. E. Solak, R. Murray-Smith, W. E. Leithead, D. J. Leith, C. E. Rasmussen. "Derivative
   observations in Gaussian process models of dynamic systems." *NeurIPS* 16, 2003.
2. B. Calderhead, M. Girolami, N. D. Lawrence. "Accelerating Bayesian inference over nonlinear
   differential equations with Gaussian processes." *NeurIPS* 21, 2008.
3. P. Wenk, A. Gotovos, S. Bauer, N. S. Gorbach, A. Krause, J. M. Buhmann. "Fast Gaussian Process
   Based Gradient Matching for Parameter Identification in Systems of Nonlinear ODEs." *AISTATS*,
   2019.
4. C. E. Rasmussen, C. K. I. Williams. *Gaussian Processes for Machine Learning*. MIT Press, 2006,
   §9.4 (odvodi GP-jev).
