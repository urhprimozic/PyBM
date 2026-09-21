# MCMC ocenjevanje parametrov ODE: kako deluje in kako se primerja z našim pristopom

*Tretji spremljevalni dokument k `integral-matching.md`/`gradient-matching.md` - tokrat o TEM, KAKO
Dondelinger et al. (2013, AGM) in Wenk et al. (2019, FGPGM) dejansko poiščeta $\theta$: z Markov Chain
Monte Carlo (MCMC), ne s `scipy.optimize.least_squares` kot mi. Cilj: natančen mehanizem MCMC, kako
ga oba članka konkretno uporabita na tem problemu, in pošten prevod v "kdaj bi nam dejansko
pomagal".*

---

## Notacija

Deljena z `gradient-matching.md`. Novo tukaj:

- $\pi(\cdot)$ - ciljna (nenormalizirana) gostota, iz katere hočemo vzorčiti,
- $q(x'|x)$ - predlagalna (proposal) gostota - verjetnost, da iz stanja $x$ predlagamo premik na $x'$,
- $\alpha(x'|x)$ - verjetnost sprejema predlaganega premika,
- $\theta^{(s)}$ - $s$-ti vzorec verige (ne točkovna ocena - CELA zaporedje vzorcev),
- PSRF - potential scale reduction factor (Gelman & Rubin 1992), diagnostika konvergence.

---

## 1. Gradniki: kako MCMC sploh deluje

### 1.1 Problem, ki ga MCMC rešuje

Imamo ciljno gostoto $\pi(\theta)$ (npr. posterior $p(\theta|y)$), ki jo znamo OVREDNOTITI v
poljubni točki (do normalizacijske konstante), ne znamo pa iz nje neposredno VZORČITI (ni standardna
oblika - Gaussovka, Gamma... - iz katere bi znal generator naključnih števil izvleči vzorec direktno).
MCMC to reši posredno: zgradi Markovsko verigo $\theta^{(0)},\theta^{(1)},\theta^{(2)},\dots$, katere
**stacionarna porazdelitev je natanko $\pi$** - če verigo poganjaš dovolj dolgo, njeni vzorci
(po "burn-in" fazi) postanejo (odvisni, a) vzorci iz $\pi$.

> **Vir:** C. P. Robert, G. Casella, *Monte Carlo Statistical Methods*, 2. izd., Springer, 2004,
> pogl. 6-7 - standarden učbeniški vir za MCMC teorijo.

### 1.2 Metropolis-Hastings algoritem

**Definicija 1.2.** Dana ciljna gostota $\pi$ (nenormalizirana) in predlagalna gostota $q(\cdot|\theta)$.
Algoritem, v vsakem koraku $t$:

1. Predlagaj $\theta'\sim q(\cdot|\theta^{(t)})$.
2. Izračunaj razmerje sprejema

$$
\alpha(\theta'|\theta^{(t)}) = \min\left(1,\; \frac{\pi(\theta')\,q(\theta^{(t)}|\theta')}{\pi(\theta^{(t)})\,q(\theta'|\theta^{(t)})}\right). \tag{1.1}
$$

3. S verjetnostjo $\alpha$ sprejmi: $\theta^{(t+1)}=\theta'$. Sicer zavrni: $\theta^{(t+1)}=\theta^{(t)}$.

**Zakaj to deluje (skica dokaza, detailed balance).** Veriga ima $\pi$ za stacionarno porazdelitev,
če zadošča pogoju **podrobnega ravnovesja** (detailed balance):

$$
\pi(\theta)\,q(\theta'|\theta)\,\alpha(\theta'|\theta) \;=\; \pi(\theta')\,q(\theta|\theta')\,\alpha(\theta|\theta'). \tag{1.2}
$$

Vstavimo (1.1) v (1.2) - za primer, ko je desna stran manjša (torej $\alpha(\theta'|\theta)=1$,
$\alpha(\theta|\theta')=\pi(\theta)q(\theta'|\theta)/[\pi(\theta')q(\theta|\theta')]$):

$$
\pi(\theta)q(\theta'|\theta)\cdot1 = \pi(\theta')q(\theta|\theta')\cdot\frac{\pi(\theta)q(\theta'|\theta)}{\pi(\theta')q(\theta|\theta')} = \pi(\theta)q(\theta'|\theta). \quad\checkmark
$$

Enakost drži identično - (1.2) je zadoščeno KONSTRUKCIJSKO, po definiciji $\alpha$ same, ne kot
dodatna predpostavka. Podrobno ravnovesje implicira stacionarnost $\pi$ (standarden rezultat za
Markovske verige - integriraj (1.2) po $\theta$, dobiš ravnotežno enačbo $\int\pi(\theta)P(\theta'|\theta)d\theta=\pi(\theta')$).

> **Vir:** W. K. Hastings, *Monte Carlo sampling methods using Markov chains and their applications*,
> Biometrika 57, 1970, 97-109 - izvorni članek.

### 1.3 Random-walk Metropolis (poseben, a najpogostejši primer)

**Definicija 1.3.** Če je $q$ SIMETRIČNA ($q(\theta'|\theta)=q(\theta|\theta')$ - npr.
$\theta'=\theta+\epsilon$, $\epsilon\sim\mathcal N(0,\sigma_p^2)$), se $q$-razmerje v (1.1) POKRAJŠA:

$$
\alpha(\theta'|\theta) = \min\left(1,\; \frac{\pi(\theta')}{\pi(\theta)}\right). \tag{1.3}
$$

To je NATANKO shema, ki jo uporabita OBA članka (Dondelinger §3, Wenk Algorithm 1: "adding a
zero mean Gaussian increment") - preprosta Gaussova motnja trenutnega stanja, sprejem/zavrnitev
glede na golo RAZMERJE ciljnih gostot.

### 1.4 Optimalna stopnja sprejema

**Trditev (Gelman, 1997).** Za random-walk Metropolis na (dovolj gladki, visokodimenzionalni)
ciljni gostoti je asimptotsko optimalna stopnja sprejema $\approx0.234$ - prenizka stopnja
(predlogi preredko sprejeti) pomeni predloge prevelike korake (veriga se komaj premakne, ko sprejme);
previsoka (skoraj vedno sprejeto) pomeni prekratke korake (veriga raziskuje prostor prepočasi).
Oba članka temu sledita: Dondelinger uravnava širine predlogov, da doseže sprejem $\approx0.25$
(dovolj blizu 0.234, njihova lastna izbira).

> **Vir:** A. Gelman, W. R. Gilks, G. O. Roberts, *Weak convergence and optimal scaling of random
> walk Metropolis algorithms*, Annals of Applied Probability 7(1), 1997, 110-120.

### 1.5 Diagnostika konvergence: PSRF (Gelman-Rubin)

**Problem.** MCMC verigi NIKOLI ni zagotovljeno, da je konvergirala v končnem času - lahko je
"obtičala" v enem delu prostora, ne da bi sploh vedela, da obstaja drug del. En sam tek ti tega ne
pove.

**Rešitev (Gelman & Rubin, 1992).** Poženi $M$ neodvisnih verig z RAZLIČNIMI začetnimi točkami.
Za vsak parameter primerjaj varianco ZNOTRAJ posamezne verige ($W$) proti varianci MED verigami
($B$) - če so verige res vse konvergirale k isti $\pi$, bi morali biti primerljivi. PSRF (tudi
$\hat R$) je grobo $\sqrt{(\text{ocenjena skupna varianca})/W}$; PSRF blizu $1$ pomeni "verige se
strinjajo", PSRF $\gg1$ pomeni "verige NISO konvergirale (ali obtičale v različnih predelih)".
Oba članka uporabita prag PSRF $<1.1$ za "dovolj konvergirano" - Dondelinger EKSPLICITNO poroča, da
Calderhead et al.-ov osnovni model NIKOLI ni dosegel PSRF $<1.1$ pri neničelnem šumu (glej §5 tam)
- to je bil GLAVNI empirični dokaz njihove diagnoze "ta metoda je nezanesljiva pod šumom".

> **Vir:** A. Gelman, D. B. Rubin, *Inference from iterative simulation using multiple sequences*,
> Statistical Science 7(4), 1992, 457-472.

### 1.6 Population MCMC / vzporedno temperiranje

**Problem, ki ga rešuje.** Random-walk Metropolis lahko obtiči v enem lokalnem vrhu (§3 spodaj -
ista skrb kot pri nas glede večkratnih lokalnih minimumov), če je med njim in drugim vrhom dolina
prenizke verjetnosti, da bi jo naključni sprehod kdaj sam prečkal.

**Rešitev (Jasra et al., 2007).** Poganjaj VEČ verig HKRATI, vsaka na "ohlajeni" različici ciljne
gostote $\pi_k(\theta)\propto\pi(\theta)^{1/T_k}$ z različno "temperaturo" $T_k$ ($T_1=1$ prava
gostota, $T_k>1$ zgladi/sploščí gostoto - lažje preskakuje med vrhovi). Občasno predlagaj IZMENJAVO
stanj med sosednjima verigama (sprejeto/zavrnjeno spet po Metropolis-Hastingsovem pravilu (1.1), na
razširjeni ciljni gostoti čez vse temperature) - vroča veriga (visok $T$) lahko "prenese" dober
kandidat hladni verigi, ki bi ga sama nikoli našla. Dondelinger uporabi TOČNO to: 10 verig pri
različnih temperaturah, eksponentna lestvica temperatur, uravnana med "burn-in" fazo za stopnjo
sprejema izmenjav $\approx0.25$.

> **Vir:** A. Jasra, D. A. Stephens, C. C. Holmes, *On population-based simulation for static
> inference*, Statistics and Computing 17(3), 2007, 263-279.

### 1.7 "Beljenje" (whitening) za sklopljene latentne spremenljivke

**Problem.** Stanja $X$ (GP-jevi vzorci) in hiperparametri $\varphi$ (lengthscale itd.) so MOČNO
sklopljeni - $X\sim\mathcal N(0,C_\varphi)$, torej sprememba $\varphi$ TAKOJ spremeni, katere $X$
sploh imajo smisel. Če bi $X$ in $\varphi$ vzorčil izmenično (Gibbs), bi vsak korak samo "podrl"
prejšnjega - zelo počasno mešanje.

**Rešitev (Murray & Adams, 2010).** Reparametriziraj: uvedi NEODVISNO $\nu\sim\mathcal N(0,I)$ in
piši $X=L_{C_\varphi}\nu$, kjer $L_{C_\varphi}L_{C_\varphi}^\top=C_\varphi$ (Choleskyjev faktor -
ISTA konstrukcija kot naš `_FittedGP.L`!). Ker sta $\nu$ in $\varphi$ SAMA PO SEBI neodvisna
(apriorno), lahko $\varphi$ vzorčiš pri FIKSNEM $\nu$ (ne fiksnem $X$) - premik $\varphi$ takrat
avtomatsko "povleče" tudi $X$ (prek $X=L_{C_\varphi}\nu$) na smiseln nov kraj, brez potrebe, da bi
morali oba usklajevati ročno.

> **Vir:** I. Murray, R. P. Adams, *Slice sampling covariance hyperparameters of latent Gaussian
> models*, NeurIPS 23, 2010.

---

## 2. Uporaba na problemu ocenjevanja parametrov ODE

### 2.1 Ciljna gostota

Oba članka vzorčita iz (variacije) iste skupne gostote - ta je NATANKO tista, ki jo Wenk et al.
(2019) matematično korektno izpelje kot Theorem 1 (glej `notes/fgpgm.md`-primerljivo diskusijo
zgoraj v chatu):

$$
p(x,\theta\mid y,\varphi,\gamma,\sigma) \;\propto\; \underbrace{p(\theta)}_{\text{prior na }\theta}
\cdot \underbrace{\mathcal N(x\mid0,C_\varphi)}_{\text{GP prior na stanje}}
\cdot \underbrace{\mathcal N(y\mid x,\sigma^2I)}_{\text{ujemanje s podatki}}
\cdot \underbrace{\mathcal N(f(x,\theta)\mid Dx,\,A+\gamma I)}_{\text{gradient-matching člen}}. \tag{2.1}
$$

Zadnji člen je NATANKO naš residual (2.5) iz `gradient-matching.md` - $Dx$ je GP-jevo posteriorno
povprečje ODVODA (naš $\hat x'(t)$), $f(x,\theta)$ je ODE-jeva desna stran $F$, $A$ pa GP-jeva
posteriorna kovarianca odvoda (naš `_FittedGP.var_deriv`, le polna kovariančna matrika, ne le
diagonala) - $\gamma I$ pa je ODE-stranska toleranca na neujemanje, ki je MI NIMAMO (glej prejšnjo
primerjavo v chatu).

### 2.2 AGM (Dondelinger et al.): dvoblokovna Gibbsova shema + populacijski MCMC

Ne vzorčijo (2.1) v ENEM koraku - razdelijo na BLOKE (Gibbsova shema, §2.1 njihovega članka):

1. $\varphi,\sigma \sim p^*(\varphi,\sigma|y)$ - GP hiperparametri, SVOJ MCMC, SAMO iz podatkov $y$
   (brez ODE) - to je Calderhead et al.-ov korak, ki ga AGM kritizira kot "brez povratne zanke", a
   ga vseeno obdrži kot PRVI blok.
2. $X\sim p(X|y,\sigma,\varphi)$ - stanja, vzorčena iz standardnega GP posteriorja (analitično znana
   Gaussovka - direktno vzorčenje, ni potreben M-H).
3. $\theta,\gamma\sim p(\theta,\gamma|X,\varphi,\sigma)$ - TO je blok, kjer AGM popravi Calderhead et
   al.: namesto da bi $\theta,\gamma$ NIKOLI vplivala nazaj na $\varphi,\sigma,X$ (enosmeren tok),
   AGM to zanko zapre (§2.2 njihovega članka) - Metropolis-Hastings na (2.1) (brez $y$-člena
   eksplicitno, ker je $X$ že fiksiran iz koraka 2).

Vse to teče znotraj POPULACIJSKEGA MCMC-ja (§1.6 zgoraj): 10 verig, temperature, izmenjalni koraki,
plus beljenje (§1.7) za sklopljenost $X,\varphi$. Konvergenca: PSRF $<1.1$ - pogosto NI dosežena
(Calderhead-ov model nikoli, AGM sam pogosto potrebuje deset tisoče korakov).

### 2.3 FGPGM (Wenk et al.): zaporeden fit + ENA veriga

Bistveno preprostejše (Algorithm 1 v njihovem članku):

1. **Korak 1 (enkrat, pred MCMC-jem):** standardiziraj podatke ($\tilde y=(y-\mu_y)/\sigma_y$ - to
   MI NE delamo), nato max-likelihood fit $\varphi_k,\sigma_k$ SAMO iz $\tilde y$ (isto kot naš
   `_fit_gp_1d`) - $\varphi,\sigma$ ostaneta FIKSIRANA za ves preostanek postopka (nobene povratne
   zanke nazaj vanju - to je NAMERNO, glej §2.3 spodaj).
2. **Korak 2:** ENA veriga Metropolis-Hastings, vzorči $x$ IN $\theta$ SKUPAJ (po komponentah - za
   vsako stanje/parameter posebej predlagaj Gaussovo motnjo, sprejmi/zavrni po (1.3) na (2.1), z
   $\varphi,\sigma$ fiksnima iz koraka 1 in $\gamma$ ROČNO nastavljenim (preizkusijo 8
   logaritemsko-razporejenih vrednosti med $1$ in $10^{-4}$, izberejo po prileganju).
3. Zavrzi "burn-in" vzorce, vrni POVPREČJE preostalih ($\mathcal S$) kot točkovno oceno - čeprav je
   cel postopek Bayesovski/MCMC, je KONČNI rezultat, ki ga vrneta, spet ena točka (povprečje
   posteriorja) - poln nabor vzorcev uporabijo samo za risanje negotovostnih pasov (Fig. 3, 5, 7-11
   v njihovem članku).

### 2.4 Ključna, presenetljiva ugotovitev (§4.1 FGPGM članka)

To je bilo že omenjeno v chatu, tu formalno zapisano: FGPGM-jeva primerjava (MVGM = variacijska
metoda + ločen max-likelihood fit hiperparametrov, PROTI AGM = skupno vzorčenje $\varphi,X,\theta$)
pokaže, da SKUPNA optimizacija $\varphi$ z ostalim (AGM-jev glavni prispevek, §2.2 zgoraj) NI
koristna za natančnost - nasprotno, poslabša jo. FGPGM zato $\varphi,\sigma$ eksplicitno ZAKLENE po
koraku 1, nikoli ju ne pusti nazaj vplivati z $\theta$ - to je NAMERNA arhitekturna odločitev, ne
pomanjkljivost. To je razlog, da je FGPGM HKRATI hitrejši (35 % manj časa, brez potrebe po
prekalkulaciji $C_\varphi$ ob vsakem koraku) IN natančnejši (do 62 % manjši RMSE) od AGM.

---

## 3. Primerjava z našim pristopom

### 3.1 Kaj je "enakovredno" našemu koraku

Naš `estimate_gradient_matching`/`estimate_integral_matching` = FGPGM-jev KORAK 1 (zaporeden
max-likelihood GP fit) + FGPGM-jev cilj (2.1)-jev zadnji člen (brez $\gamma$, brez $y$-člena, ker
mi $X$ NIKOLI ne pustimo prosto - vedno je fiksiran na $\hat x(t)$) - MINUS korak 2 (MCMC). Namesto
MCMC-ja nad $(x,\theta)$ mi ENKRAT poiščemo VRH z `least_squares`, in $x$ NIKOLI ne postane prosta
spremenljivka (glej prejšnjo primerjavo v chatu - to je bila naša glavna strukturna razlika od
FGPGM/Ramsay et al.).

### 3.2 Tri stvari, ki jih MCMC reši, mi pa jih rešujemo (ali ne rešujemo) drugače

Neposredna povezava na prejšnji del pogovora (trije LOČENI problemi lokalne Gaussove/kvadratične
aproksimacije):

1. **Robne rešitve** (konstanta pripeta na mejo, npr. `refTemp=22.0`): MCMC na (2.1) TOČNO
   reprezentira robno omejitev (Metropolis-Hastings predlog, ki bi šel čez mejo, se v `π`
   preprosto ovrednoti na 0 - zavrnjen), zato vzorci pravilno pokažejo NESIMETRIČNO gostoto ob robu
   - naša Laplaceova aproksimacija to napačno simetrizira. **MCMC je tu genuinely boljši.**
2. **Ozke doline/skoraj-singularne smeri**: MCMC vzorci sledijo PRAVI (lahko ukrivljeni, ne nujno
   eliptični) obliki doline, kolikor daleč jo raziščejo - naša $G^{-1}$ elipsa je samo LOKALNA
   (2. red) aproksimacija okoli vrha, ki lahko zgreši, kako se dolina dejansko ukrivlja dlje stran.
   **MCMC je spet boljši, a dražje.**
3. **Prave večkratne lokalne minimume**: TU je populacijski MCMC (§1.6 zgoraj) SPECIFIČNO
   zasnovan za to (temperature, izmenjalni koraki eksplicitno pomagajo ubežati enemu vrhu). Naš
   `least_squares` (en sam tek, en `c0`) nima NOBENE zaščite - to smo v chatu že priznali kot
   nepreverjeno vrzel.

### 3.3 Cena

Dondelingerjev članek sam poroča red velikosti: $10^5$ MCMC korakov traja $\sim$1000-15000 sekund
(odvisno od metode, njihova Fig. 4), konvergenca (PSRF$<1.1$) pogosto rabi deset tisoče korakov (ali
je sploh NE doseže - Calderhead-ov model nikoli). To je REDI VELIKOSTI dražje od našega ENEGA
`least_squares` teka (~15-30 evaluacij do konvergence, ~25s na Bled podatkih - glej prejšnje
profiliranje v chatu).

### 3.4 Primerjalna tabela

| | AGM (Dondelinger) | FGPGM (Wenk) | Mi (`pybm.estimate.*`) |
|---|---|---|---|
| Kaj vrne | vzorci $(x,\theta)$ | vzorci, POVPREČJE kot točka | ena točka $\hat c$ |
| GP hiperparametri $\varphi$ | vzorčeni SKUPAJ z $\theta$ | fiksirani PRED $\theta$ (§2.4 - namerno!) | fiksirani pred $\theta$ |
| $x$ (stanje) med fitanjem $\theta$ | prosto (vzorčeno) | prosto (vzorčeno) | FIKSIRANO na $\hat x(t)$ |
| $\gamma$ (ODE-GP toleranca) | vzorčen | ročno pomeden (8 vrednosti) | nimamo ga |
| Robne rešitve | pravilno predstavljene | pravilno predstavljene | Laplaceova aprox. jih napačno simetrizira |
| Večkratni vrhovi | populacijski MCMC ublaži | NE ublaži (ena veriga) | NE ublaži (en `least_squares` tek) |
| Cena na fit | zelo visoka ($10^3$-$10^4$s+) | nižja od AGM (-35%), a še vedno MCMC | nizka (~10-30s) |
| Standardizacija podatkov pred GP fitom | ni omenjena | DA (z-score) | **NE** (poceni popravek, ki bi ga lahko dodali) |

---

## 4. Povzetek

1. MCMC/Metropolis-Hastings ni magija - je konstrukcija verige, katere STACIONARNA porazdelitev je
   (dokazano, prek podrobnega ravnovesja) natanko ciljna gostota, iz katere je težko vzorčiti
   direktno.
2. AGM in FGPGM vzorčita PRIBLIŽNO isto skupno gostoto (2.1), a se RAZLIKUJETA v tem, ali $\varphi$
   (GP hiperparametre) vzorčita SKUPAJ s $\theta$ (AGM) ali FIKSIRATA PRED (FGPGM) - FGPGM-jeva
   empirična ugotovitev je, da je fiksiranje BOLJŠE, kar posredno VALIDIRA arhitekturo, ki jo mi že
   uporabljamo (zaporedno, ne skupno).
3. Kar mi NIMAMO in bi lahko poceni dodali: standardizacija podatkov pred GP fitom (FGPGM), $\gamma$
   kot dodaten prost parameter v uteži (glej prejšnjo primerjavo `weight_by_uncertainty`).
4. Kar mi NIMAMO in bi bilo DRAGO dodati: prosto (ne fiksirano) stanje $x$ med fitanjem, in kakršnokoli
   zaščito pred večkratnimi lokalnimi vrhovi (MCMC/populacijski MCMC ali vsaj multi-start).
5. Cena razlike je REDI VELIKOSTI - MCMC dá popolnejšo sliko negotovosti (robovi, ukrivljene doline,
   večmodalnost), a za ceno $10^2$-$10^3$x počasnejšega fita. Za našo rabo (hitro iterativno
   primerjanje čez 8 foldov, več struktur, več metod) je to trenutno slab kompromis - za KONČNO,
   skrbno analizo ENEGA že izbranega modela pa bi bil MCMC pravi naslednji korak.

---

## Literatura

Novi viri, specifični za ta dokument (deljeni viri glej `integral-matching.md`/`gradient-matching.md`):

1. C. P. Robert, G. Casella. *Monte Carlo Statistical Methods*, 2nd ed. Springer, 2004.
2. W. K. Hastings. "Monte Carlo sampling methods using Markov chains and their applications."
   *Biometrika* 57, 1970, 97-109.
3. A. Gelman, W. R. Gilks, G. O. Roberts. "Weak convergence and optimal scaling of random walk
   Metropolis algorithms." *Annals of Applied Probability* 7(1), 1997, 110-120.
4. A. Gelman, D. B. Rubin. "Inference from iterative simulation using multiple sequences."
   *Statistical Science* 7(4), 1992, 457-472.
5. A. Jasra, D. A. Stephens, C. C. Holmes. "On population-based simulation for static inference."
   *Statistics and Computing* 17(3), 2007, 263-279.
6. I. Murray, R. P. Adams. "Slice sampling covariance hyperparameters of latent Gaussian models."
   *NeurIPS* 23, 2010.
7. F. Dondelinger, M. Filippone, S. Rogers, D. Husmeier. "ODE parameter inference using adaptive
   gradient matching with Gaussian processes." *AISTATS*, 2013.
8. P. Wenk, A. Gotovos, S. Bauer, N. S. Gorbach, A. Krause, J. M. Buhmann. "Fast Gaussian process
   based gradient matching for parameter identification in systems of nonlinear ODEs." *AISTATS*, 2019.
