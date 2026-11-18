# 3. Who owns the heat loop?

Asked by brownfield-compute (2026-11-16). Heat is metered and billed, so this isn't a
free-rider problem. The risk is one operator controlling heat that people's homes depend on.
**Consumer co-op, municipal, or operator-owned: what does each do to prices, trust and upkeep?
How is the heat price set and capped? What happens to customers when the heat source, here the
data center, shuts or leaves?**

Evidence here is real cases and research, not the simulation. Brownfield's own model found the
heat business loses money on its own, so the operator funding the network in the community
benefits agreement is the likely shape. This answer is about who owns and runs it after it's
built.

## Short answer

The data center should **sell** heat into the network, not **own** the network. The pipes, the
heat pumps and the backup boilers belong to the customers (a co-op) or the public (a municipal
authority). Prices follow a cost-based rule that's published every year. And the network keeps
enough backup to heat every home without the data center.

## What happens, and the cases behind it

1. **Denmark: customer- and town-owned, at cost.** Since the Heat Supply Act (1979), Danish
   district heating runs on a non-profit rule: prices cover costs, no more and no less. There are
   about 323 consumer co-ops, and co-ops plus municipal companies supply about 94% of the heat (a
   trade-association figure, to check). The regulator, Forsyningstilsynet, publishes every
   network's prices at least once a year.
2. **Sweden: selling off the networks raised prices.** After deregulation in 1996, many towns sold
   their energy companies. Municipal companies' share of heat fell from 98% (1990) to 65% (2002).
   Prices rose quickly, especially in Stockholm and Uppsala, and the protests led to two
   government inquiries. Two economics studies find private networks charge more than municipal
   ones, about 7% on average, mostly in the **fixed** part of the bill (Lundin, IFN working paper
   1532, 2025; Egüez, Umeå working paper 980).
3. **UK: a monopoly needs a referee.** Heat-network customers can't switch supplier. The
   competition authority found many networks priced about like gas, but said nothing guaranteed it
   and called for regulation (CMA 2018). Ofgem became the regulator on 27 January 2026. It can
   investigate profits that look disproportionate, and it can direct another operator to step in
   and keep the heat on if one fails ("last resort directions").
4. **Pennsylvania already regulates a heat utility.** Philadelphia's steam loop (Vicinity Energy)
   is under the Public Utility Commission: it files a formula-based steam cost rate every month,
   and the commission audits it. A network selling heat to Homer City homes would probably be
   regulated too (to check with a Pennsylvania utility lawyer, including whether a co-op or
   municipal authority is treated differently).
5. **Data centers already sell heat to town-owned networks.**
   - **Odense, Denmark:** Meta's data center heat goes into Fjernvarme Fyn's network, and Fyn
     owns the heat pumps that lift it to 70-75 °C. Fyn says it's equal to about 11,000
     households (2025).
   - **Mäntsälä, Finland:** the Yandex (now Nebius) data center feeds the local utility Nivos's
     network. Nebius says connected homes pay about 10% less for heat (a company claim).
6. **When the data center stops.** In April 2022, the Mäntsälä data center lost its grid power
   in a contract dispute. It ran on diesel at reduced load and put little or no heat into the
   network. Nivos kept the town warm with light fuel oil and natural gas. Homes stayed heated
   because the network owner had its own backup.
7. **The risk is real but smaller than feared.** A study of 107 industrial waste-heat deals in
   Sweden found about 6% of the recovered heat was lost: a plant closed, or the heat was replaced
   by another source. The authors say feasibility studies usually overestimate this risk (Werner &
   Lygnerud 2018, Energy 151). A Norwegian operator describes the standard rule: keep enough
   backup to supply every customer if the largest heat source fails.

## What it would change in the design

- **The operator pays to build the network, then a customer co-op or a municipal or joint
  authority owns it.** The data center is one seller of heat. (In answer 2 the operator funds
  the shared places and a trust runs them; this is the same pattern.)
- **A cost-based price rule** in the co-op's bylaws or the benefits agreement: prices cover
  costs, published every year, with a reserve for repairs and replacement.
- **Watch the fixed charge.** In Sweden that's where private networks charged more. Cap it, and
  publish how it's built up.
- **A price ceiling against the alternative:** a home on the loop should never pay more over a
  year than with its own efficient heat pump (brownfield's comparison case). This is logic, not
  a tested rule; Swedish companies argue heat pumps are what keep them honest.
- **Backup sized to heat every home without the data center**, owned by the network. Brownfield
  costs this.
- **A heat purchase contract with the data center:** long term, a notice period before it can
  stop, and a guarantee or bond that pays for backup heat if it leaves early. It should sell
  heat cheap, since taking the heat saves the data center cooling.
- **Name a step-in party** in the agreement (like Ofgem's last-resort directions), so someone
  has to keep the heat on if the network operator fails.
- **Expect Pennsylvania utility regulation**, and design so the commission sees a cost-based
  utility, not a profit centre.

## How sure we are

**Medium to high** on the direction (customer or town ownership, cost-based prices, backup):
- the Danish model has run for about 45 years, though most of the sources promoting it are
  Danish industry bodies;
- two economics studies from Sweden point the same way;
- the UK regulator's reasoning is about exactly this kind of monopoly;
- the backup lesson rests on one real failure (Mäntsälä) and a 107-case study.

**Low** on the details: the price ceiling, how a co-op would be regulated in Pennsylvania, and
any figures that come only from companies.

## Where Melting Pot could help later

This is a power question more than a cleaning one. A game where one player controls something
everyone needs (Territory, or a custom one) could show how often and how fast the owner starts
squeezing, and which rules stop it: published prices, a ceiling, a customer vote. It would point
at cases to check, not replace them.

## Sources

- Danish model: Danish District Heating Association. https://danskfjernvarme.dk/uk/english/the-danish-model ;
  Aalborg University grassroots project. https://grassroots.aau.dk/challenge-denmark/ ;
  Life in Denmark (heating for consumers). https://lifeindenmark.borger.dk/housing-and-moving/gas--electricity--heating-and-water-supply/heating-for-consumers
- Sweden: Lund University chapter. https://lup.lub.lu.se/record/5e79091d-aeeb-47e3-b1d7-1db53716b61b ;
  Lundin, IFN WP 1532 (2025). https://ifn.se/en/publications/working-papers/2025/1532/ ;
  Egüez, Umeå WP 980. https://swopec.hhs.se/umnees/abs/umnees0980.htm
- UK: CMA heat networks study coverage. https://www.businessgreen.com/news/3032105/heat-networks-regulation-now-needed-says-uk-competition-watchdog ;
  Ofgem price protections draft decision (2026). https://consult.ofgem.gov.uk/energy-supply/heat-networks-regulation-fair-pricing-protection/supporting_documents/heat-networks-regulation-price-protections-draft-guidance-decisionpdf ;
  Trowers & Hamlins. https://www.trowers.com/insights/fair-pricing-protections-what-does-this-mean-for-heat-networks
- Pennsylvania: PUC docket M-2025-3055974 (Vicinity steam cost rate). https://www.puc.pa.gov/docket/M-2025-3055974
- Odense: Ramboll. https://ramboll.com/en-gb/projects/energy/meta-surplus-heat-to-district-heating ;
  Meta 2018. https://tech.facebook.com/engineering/2018/4/odense-dc
- Mäntsälä: DCD on the power cut. https://datacenterdynamics.com/en/news/yandex-data-center-in-finland-loses-power-runs-on-diesel-due-to-war-sanctions ;
  The Register. https://www.theregister.com/2022/05/05/yandex_datacenter_in_finland/ ;
  Nebius expansion. https://datacenterdynamics.com/en/news/nebius-to-triple-capacity-of-mantsala-data-center-in-finland
- Werner & Lygnerud 2018, "Risk of industrial heat recovery in district heating systems",
  Energy 151. https://ideas.repec.org/a/eee/energy/v151y2018icp430-441.html
- Backup rule: Norwegian operator statement. https://www.mip.no/en/wp-json/wp/v2/posts/40430
