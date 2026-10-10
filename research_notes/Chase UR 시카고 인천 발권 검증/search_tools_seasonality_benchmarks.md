# Award search tools, release patterns, seasonality, ORD→ICN winter 2026/27 schedule, and cash-fare benchmarks (as of 2026-10-10, for travel 2026-11-20 → 2026-12-10)

*Research constraints (read first):* In this session every direct page fetch (WebFetch and curl through the egress proxy) was refused by organization egress policy — 30 hosts probed (frequentmiler.com, aeroroutes.com, seats.aero, flyasiana.com, koreanair.com, kayak.com, google.com/travel/flights, reddit.com, flyertalk.com, milemoa.com, etc.) all returned 403 at CONNECT — and the shared WebSearch budget was exhausted after ~35 queries. **Every finding below therefore comes from search-engine result summaries/snippets, not from reading the full page.** Numbers are reproduced as the snippets stated them; dates are given where the snippet exposed one, otherwise marked "(undated in snippet)". Anything older than mid-2025 is flagged **POTENTIALLY OUTDATED**; anything not corroborated is flagged **UNVERIFIED**. Blog/forum/broker content is labeled **anecdotal**. Google Flights, Kayak route pages for ORD–ICN, Reddit r/awardtravel and FlyerTalk could not be consulted at all.

---

## KQ1. Which award-search tools cover which programs (Virgin Atlantic, Aeroplan, United, Flying Blue, BA, Qatar, KrisFlyer), pricing, alerts, freshness, blind spots

### Takeaway
As of 2026 reviews, Roame is the only tool whose own listing explicitly names all of Flying Blue, Virgin Atlantic Flying Club, Aeroplan, Qatar Privilege Club and KrisFlyer with live (real-time) searches; Seats.aero is the cheapest/strongest for wide date-range scanning and alerts but is cached data (reviews warn results can be stale); PointsYeah is the strongest free tier (Virgin, Flying Blue, Aeroplan, KrisFlyer confirmed; Qatar not confirmed); point.me is the hand-holding/booking-instructions option with the broadest program count (30+) but the highest price. 한 줄 요약: 넓게 훑기·알림은 Seats.aero(캐시), 실시간 확인은 Roame/PointsYeah, 발권 가이드가 필요하면 point.me.

### Cited Findings

**Cross-tool comparisons (2026)**
- Pricing snapshot in a 2026 round-up: Seats.aero "Free / Pro $9.99 mo or $99.99 yr"; Roame "Friends tier from $12.99 mo or $109.99 yr"; point.me "Standard Annual $10.75 mo billed yearly" (undated 2026 article) — [The Points Party, "Best Award Search Engines in 2026"](https://thepointsparty.com/articles/best-award-search-engines)
- Same article's use-case verdicts: Lufthansa First → Seats.aero; Aeroplan partner awards → PointsYeah or point.me; BA Avios sweet spots → Roame; SkyTeam routings → point.me, with Seats.aero described as "uneven" for SkyTeam — [The Points Party](https://thepointsparty.com/articles/best-award-search-engines)
- Program breadth: "Point.me has 30+ programs included in its search engine, while Seats.aero and Roame have around 15" (undated in snippet) — [Milesopedia comparison](https://milesopedia.com/en/reviews/products/comparison-best-points-flight-search-tools/)
- Free-tier date flexibility: Seats.aero free search is "Fixed, up to 60 days out"; PointsYeah free date range "Flexible +/- 3 days" (2026) — [The Miles Market, PointsYeah vs Seats.aero 2026](https://www.themilesmarket.com/post/pointsyeah-vs-seats-aero-the-ultimate-award-search-tool-review)
- Qualitative verdicts (2026): Seats.aero "great for scanning huge date ranges and finding products like Qsuites or Lufthansa First" but "data can be seriously outdated"; point.me "great hand-holding and step-by-step booking instructions, but slower and more expensive"; Roame "polished interface… particularly strong on Avios and OneWorld partner space" — [Feisworld, "Best Award Flight Search Tools in 2026"](https://www.feisworld.com/blog/best-award-flight-search-tools)
- **UNVERIFIED** claim in the same article that PointsYeah was named NerdWallet's best award search tool of 2026 over Seats.aero, Roame and point.me — [Feisworld](https://www.feisworld.com/blog/best-award-flight-search-tools); NerdWallet's own PointsYeah review exists at [NerdWallet](https://www.nerdwallet.com/article/travel/points-yeah-award-search-review-easily-find-your-next-points-redemption) (could not be read)
- Other comparison pages located but not readable in this session (fetch blocked): [Frequent Miler, "Which award search tool is best?"](https://frequentmiler.com/which-award-search-tool-is-best/); [Frugal Flyer tool comparison](https://frugalflyer.ca/blog/award-flight-search-tool-comparison/); competitor-authored (bias risk) [AwardTool "Best Roame alternatives"](https://www.awardtool.com/compare/best-roame-alternatives) and [AwardTool "Best point.me alternatives"](https://www.awardtool.com/compare/best-point-me-alternatives)

**Seats.aero (cached search + alerts)**
- Mechanism: "Seats.aero periodically polls airline websites for award space and caches the results"; program list "includes both Air France/KLM Flying Blue and Virgin Atlantic Flying Club" (Nov 2024 guide — POTENTIALLY OUTDATED on exact list) — [MileLion Seats.aero guide](https://milelion.com/2024/11/24/seats-aero-search-star-alliance-oneworld-and-skyteam-award-space-easily/)
- Qatar Privilege Club and SAS were added as sources per the article title "Seats.aero… (now supports Qatar and SAS)" (undated in snippet) — [Frequent Miler](https://frequentmiler.com/seats-aero/)
- Price: "PRO account unlocks up to a year of availability on all routes, advanced filters, SMS alerts, and more. Only $9.99/mo"; annual $99.99 — [Upgraded Points](https://upgradedpoints.com/travel/airlines/seats-aero/); [One Mile at a Time guide](https://onemileatatime.com/guides/seats-aero/); [TPG guide](https://thepointsguy.com/travel/seats-aero/); official [seats.aero/about](https://seats.aero/about)
- Alerts: "Create an unlimited amount of alerts for award availability at no cost" (email); Pro adds SMS; "you need a Pro account if you want to set alerts more than 60 days in advance"; Pro alert filters: direct only, minimum seats, max points, max taxes/fees, max stops, specific fare class, specific flight number, exclude carriers — [Upgraded Points](https://upgradedpoints.com/travel/airlines/seats-aero/); [Travel-Dealz](https://travel-dealz.com/blog/seats-aero/)
- July 2026 update (per a 2026 review): one-click price-drop alerts when award costs fall; Pro members can enable cardholder pricing for United MileagePlus and Delta SkyMiles — [Kudos, "Seats.aero Review 2026"](https://www.joinkudos.com/blog/seats-aero-review-2026-the-best-tool-for-finding-award-flights)
- Seat Map Viewer is a Pro feature — [Australian Frequent Flyer forum (anecdotal)](https://www.australianfrequentflyer.com.au/community/threads/seats-aero-seat-maps.117784/)

**Roame (live search)**
- App Store listing names Flying Blue, Virgin Atlantic Flying Club, Aeroplan, Qatar Privilege Club and KrisFlyer among supported programs and advertises "real-time award search across more than 28 airline loyalty programs" — [Roame App Store listing](https://apps.apple.com/app/id6466661045)
- Live Search returns "results identical to those on airline websites, in real time", up to 365 days ahead, economy/business/first; caveats: Aeroplan results may include US-card transfer promos and do not show Aeroplan co-brand preferred pricing; taxes/fees displayed in USD; program count given as 23 — [Milesopedia Roame guide](https://milesopedia.com/en/reviews/products/roame/)
- Free "Community" tier flexible search ±1 day vs ±3 days on paid "Friends of Roame" from US$12.99/month; 21 programs counted at time of writing (Mar 2025); useful for KrisFlyer because singaporeair.com now requires a minimum mileage balance to search awards — [MileLion, Mar 4 2025](https://milelion.com/2025/03/04/roame-faster-searching-for-singapore-airlines-and-other-awards/)
- Older Prince of Travel program list omits Qatar and KrisFlyer (superseded) — [Prince of Travel](https://princeoftravel.com/?p=176163); AwardWallet review — [AwardWallet](https://awardwallet.com/blog/roame-travel-review/)

**PointsYeah (free search + free alerts)**
- Program list includes "Virgin Atlantic Flying Club", "Air France / KLM Flying Blue", Aeroplan and Avios programs — [NerdWallet PointsYeah review](https://www.nerdwallet.com/article/travel/points-yeah-award-search-review-easily-find-your-next-points-redemption)
- "Singapore Krisflyer" listed (added after earlier coverage); "22 airline loyalty programs and six hotel loyalty programs" — [PointsMath](https://pointsmath.com/?p=10199)
- Milesopedia: "25 loyalty programs (including 17 airline programs)" (undated) — [Milesopedia PointsYeah review](https://milesopedia.com/en/reviews/products/pointsyeah/)
- Free award searches and alerts, seat maps, hotels — [Frequent Miler](https://frequentmiler.com/?p=218709); [AwardWallet review](https://awardwallet.com/blog/pointsyeah-review/)
- Qatar Privilege Club: **not found in any PointsYeah coverage list** (absence of evidence, not confirmed absence)

**point.me (live search + booking instructions + concierge)**
- Plans: Basic free (limited); Standard $12/month or $129/year (real-time searches, booking instructions, deal alerts); Premium $260/year (adds points-strategy consultation, $100 annual concierge credit, 10% concierge discount); Concierge from $200 per passenger round-trip — [point.me help: services and pricing](https://connect.point.me/help/what-services-and-pricing-does-point.me-offer); [point.me help: plan differences](https://connect.point.me/help/whats-the-difference-between-the-self-serve-plans); [Upgraded Points](https://upgradedpoints.com/news/point-me-premium-plan-discount/)
- Promo codes seen: 35% off Premium ($260→$169); 40% off Standard ($77.40) — time-limited, affiliate-sourced — [Milesopedia](https://milesopedia.com/en/reviews/products/point-me-award-searching-tool/); [AwardWallet coupon page](https://awardwallet.com/travel/airlines/pointme-coupon-code/); [FlightPoints PointsYeah vs point.me 2026](https://flightpoints.com/blogs/pointsyeah-vs-point-me/)

**AwardFares**
- Has a dedicated Flying Blue program search page and guide — [AwardFares Flying Blue program page](https://awardfares.com/programs/air-france-klm-flying-blue); [AwardFares Flying Blue guide](https://awardfares.com/blog/flying-blue-guide/)
- Launched "Intelligent Alerts" in August 2025 (details not in snippet) — [AwardFares announcements](https://awardfares.com/blog/categories/announcements/)
- One AwardFares page notes that Avios's own calendar tool "excludes partner carriers like Qatar Airways" (context ambiguous in snippet) — [AwardFares blog](https://awardfares.com/blog/page/3/)
- One reviewer preferred AwardFares' depth over Roame for their workflow ("beautiful and friendly, but not as deep as AwardFares") — [Feisworld](https://www.feisworld.com/blog/best-award-flight-search-tools)

**Surfacing specific partner space**
- Korean Air via Virgin Atlantic: "partner space can be limited, especially in premium cabins on popular routes"; booking is **by phone through Virgin's Flying Club service centre**; Korean Air First is not bookable via Virgin (page dated June 2026) — [Award Travel Finder, Virgin Points → Korean Air](https://awardtravelfinder.com/how-many-virgin-points-to-fly-korean-air)
- Virgin Atlantic Flying Club described as "great for finding SkyTeam saver awards (Delta, Air France, KLM, Korean Air, etc.)" and included as a Seats.aero source (Nov 2024 — POTENTIALLY OUTDATED) — [MileLion](https://milelion.com/2024/11/24/seats-aero-search-star-alliance-oneworld-and-skyteam-award-space-easily/)
- Korean Air "O"-class partner inventory: bookable by Delta when it shows open, but Delta enforces blackout dates broadly; ExpertFlyer may show space Delta then refuses; Amex→Flying Blue cited as the alternative route to the same space (2024-era — POTENTIALLY OUTDATED) — [View from the Wing](https://viewfromthewing.com/korean-airlines-business-class-award-space-now-easier-to-find-and-how-you-can-book-it/)
- Historical Korean Air partner restriction: 2016 report of ≤2 economy partner seats per flight and ~1 business seat — [One Mile at a Time, 2016 (POTENTIALLY OUTDATED)](https://onemileatatime.com/korean-air-greatly-limiting-partner-award-availability/); 2018 report of less availability within ~2 months of departure — [TPG, 2018 (POTENTIALLY OUTDATED)](https://thepointsguy.com/news/korean-air-reduced-award-availability/)
- JAL via BA: American's own search does not show JAL partner space; "search on BA or Qantas first. If space shows at 331 days out, call AA" — [MilesMate AAdvantage Japan guide (anecdotal)](https://milesmate.io/blog/aadvantage-business-class-japan-guide)
- EVA via Aeroplan: partner business seats now appear ~7 days before departure; "routes such as Chicago (ORD) have shown up to eight seats available via Aeroplan" — [InsideFlyer](https://insideflyer.com/posts/eva-air-now-releases-business-class-seats-to-partners/) (see KQ2)

### Inferences
- For this specific trip (SkyTeam nonstop on Korean Air), the practical stack is: Seats.aero (Flying Blue + Virgin Atlantic sources, free email alerts; Pro needed for alerts beyond 60 days, which no longer matters inside a 6–8-week horizon) to catch space appearing, then Roame or the airline site to confirm live before transferring points — because Seats.aero is cached and reviewers explicitly warn of stale results.
- Virgin Atlantic cannot be relied on as a self-serve path for Korean Air (phone booking per Award Travel Finder); Flying Blue is the online SkyTeam path. Tools that scrape Flying Blue (Seats.aero, PointsYeah, Roame, point.me) should therefore be the primary Korean Air monitors.
- For Star Alliance one-stops (ANA/EVA/Asiana via Aeroplan or United), Seats.aero's Aeroplan and United sources plus Roame live checks are the natural monitors; note Roame does not show Aeroplan co-brand card pricing, and Seats.aero only shows United cardholder pricing on Pro.

### Gaps
- No official, current (Oct 2026) program-coverage list could be read for Seats.aero, PointsYeah, point.me, AwardTool or AwardFares (sites blocked); coverage of United MileagePlus and British Airways by Roame/PointsYeah, and of Qatar/KrisFlyer/Virgin by point.me, was not confirmed by any snippet.
- AwardTool: no review or pricing surfaced under that name (search returned unrelated "Award Force"/"AwardKit" products); only AwardTool's own competitor-comparison pages were found.
- AwardFares pricing tiers and its exact program list were not found.
- Data-freshness specifics (how often Seats.aero re-polls Flying Blue/Virgin/Aeroplan) were not found.
- No 2025–2026 review specifically documenting a tool blind spot for Korean Air space via Flying Blue/Virgin, or for Asiana/ANA via Aeroplan/United, was found.

---

## KQ2. Award calendar opening windows and documented last-minute release patterns (Korean Air, Asiana, ANA, JAL, EVA, Turkish, Air Canada, United, Qatar, AF-KLM)

### Takeaway
Opening windows cluster at 355–361 days (Korean Air 361, Asiana 361, EVA 360, Qatar 361, Flying Blue 359–360, ANA 355 (328 for partners), BA 355, Aeroplan 355, Turkish 354–355, United 337, Virgin Atlantic 331, AAdvantage 331) — with the rule that the booking program's shorter window governs partner awards. Documented close-in patterns relevant to a Nov–Dec 2026 trip: EVA now opens partner business space ~7 days out (2026 reports, ORD cited with up to 8 seats), JAL opens remaining seats within roughly 1–14 days (anecdotal), and no reliable Korean Air/Asiana close-in pattern was found. 한 줄 요약: 6~8주 전인 지금은 '오픈 시점'보다 '막판 풀림' 모니터링이 핵심이며, EVA(약 7일 전)·JAL(1~14일 전)은 패턴이 보고됨, 대한항공/아시아나는 근거 부족.

### Cited Findings

**Opening windows (days before departure)**
- Table (undated): Asiana 361 days; Korean Air 361 days; EVA Air 360 days; ANA 355 days (328 days for partners) — [DealsPoints release table](https://dealspoints.com/airlines-award-seat-release-time/)
- ANA: window opens 355 days out "which is when flights are added to ANA's schedule"; rule: "the frequent flyer program you're using to book dictates the award booking window — NOT the airline you fly… even if EVA Air releases seats 360 days in advance, you won't see them using United MileagePlus miles until day 337" (undated) — [AwardWallet](https://awardwallet.com/blog/far-advance-can-book-airline-award-tickets/)
- ANA official: domestic award reservations open 355 days before departure (for flights from Feb 3, 2025) — [ANA official page](https://www.ana.co.jp/en/sg/amc/domestic-flight-awards/companion-fare/)
- Qatar Airways Privilege Club: 361 days (page dated June 2026) — [Award Travel Finder, "When do airlines release award seats?"](https://awardtravelfinder.com/when-do-airlines-release-award-seats)
- Flying Blue: 360 days (Award Travel Finder) vs 359 days (other guide) vs "~330–355" general range — [Award Travel Finder](https://awardtravelfinder.com/when-do-airlines-release-award-seats); [Travel on Point(s) release dates](https://travel-on-points.com/airline-award-release-dates/); [The Points Analyst](https://www.thepointsanalyst.com/when-do-airlines-release-award-seats/)
- Virgin Atlantic Flying Club: 331 days (consistent across sources) — [Award Travel Finder](https://awardtravelfinder.com/when-do-airlines-release-award-seats); [Travel on Point(s)](https://travel-on-points.com/airline-award-release-dates/)
- Turkish Miles&Smiles: 354 days (Award Travel Finder, Travel on Point(s)) vs 355 days (The Points Analyst) — same three sources as above
- British Airways: releases "exactly 355 days before departure, at midnight London time"; the same page's claim of further waves at T-14/T-7/T-24h is **UNVERIFIED** (not corroborated elsewhere) — [Award Travel Finder BA guide (2026)](https://awardtravelfinder.com/when-does-british-airways-release-award-seats)
- Air Canada Aeroplan: "permits booking 355 days in advance" while EVA releases up to 360 days out (Apr 2025) — [Mighty Travels, Apr 2025 (anecdotal)](https://www.mightytravels.com/2025/04/sweet-spot-using-aeroplan-points-for-premium-cabin-awards-to-asia-in-2025/)
- JAL: opens 360 days out at 9 AM Japan time — [Roame JAL guide](https://roame.travel/guides/how-to-book-jal-a350-business-first-class); "typically releases one first-class and two business-class award seats at the 360-day mark… availability may be gone before the AAdvantage booking window [331 days] even opens" — [MilesMate (anecdotal)](https://milesmate.io/blog/aadvantage-business-class-japan-guide)
- Partner rule restated: "you are limited by the shorter of the two award release schedules" — [Prince of Travel, release patterns guide](https://princeoftravel.com/guides/the-most-common-award-availability-release-patterns/)
- Additional release-date guides located but unreadable: [AwardFares "Ultimate guide to award release dates"](https://awardfares.com/blog/ultimate-guide-to-award-release-dates/)

**Last-minute / close-in release patterns**
- EVA Air (2026): "Starting now, EVA Air makes business class award inventory available to partner loyalty programs like Aeroplan, typically seven days before departure. Routes such as Chicago (ORD) have shown up to eight seats available via Aeroplan." Applies to partner bookings only (July, 2026 context) — [InsideFlyer](https://insideflyer.com/posts/eva-air-now-releases-business-class-seats-to-partners/)
- Caution on EVA pattern durability: "I wouldn't bank on this continuing yet" — [View from the Wing](https://viewfromthewing.com/business-class-awards-to-asia-have-been-brutal-star-alliances-eva-air-just-opened-a-new-way-to-get-there/)
- EVA background: previously "insurmountable" partner availability; historically EVA let United/Air Canada/ANA book 2+ business seats 10+ months out (older piece — POTENTIALLY OUTDATED) — [Thrifty Traveler EVA ORD–TPE review](https://thriftytraveler.com/reviews/flights/eva-air-business-class-chicago-taipei/)
- JAL close-in: "often releases additional award availability 2 weeks or less before departure" — [Roame](https://roame.travel/guides/how-to-book-jal-a350-business-first-class); "releases all unoccupied seats available for award redemption 1 to 2 days before departure" — [Upgraded Points](https://upgradedpoints.com/travel/airlines/best-ways-to-book-japan-airlines-business-class/); reader comment: business "released consistently 1-5 days before departure" — [OMAAT community (anecdotal)](https://onemileatatime.com/community/other-airlines/how-to-redeem-miles-for-japan-airlines-a350-first-business-class/)
- ANA: no source confirmed a "~2 weeks out" pattern. One site claims "meaningful space at schedule open (~330 days out) plus additional inventory closer to departure" (low-quality source) — [BookMyBusinessClass (anecdotal)](https://bookmybusinessclass.com/miles/mileageplus/ana); an Aug 2026 article claims the "largest release wave lands in the 75–105-day window" and "availability jumps to sixty-three percent around the 90-day mark" — **UNVERIFIED**, uncorroborated — [Mighty Travels, Aug 2026](https://www.mightytravels.com/2026/08/ana-business-class-in-2025-four-paths-one-60k-mile-baseline/); practical warning that transfers can take up to 48 h and space is often gone by then — [TravelSort (anecdotal)](https://travelsort.com/ana-the-room-business-class-award-space/)
- Korean Air: no 2024–2026 source documenting a within-days partner release was found. Generic statement only: airlines "may release unsold seats for award redemptions within a week or two of departure, especially in premium cabins" — [The Points Analyst](https://www.thepointsanalyst.com/when-do-airlines-release-award-seats/); 2014 note that Korean Air First awards were "incredibly easy to find" — [Executive Traveller, 2014 (POTENTIALLY OUTDATED)](https://executivetraveller.com/which-airlines-release-last-minute-first-class-awards); 2018 finding of less availability within ~2 months — [TPG, 2018 (POTENTIALLY OUTDATED)](https://thepointsguy.com/news/korean-air-reduced-award-availability/)
- Korean Air release-window page exists (unreadable) — [Award Travel Finder, KE SkyPass](https://awardtravelfinder.com/award-release-windows/korean-air)

### Inferences
- With departure 6–8 weeks away, every program's opening window has long passed; the only levers are (a) space that is still open now and (b) close-in releases. The best-documented close-in opportunity for an ORD origin is EVA via Aeroplan (~7 days before departure, ORD explicitly cited) — which requires Aeroplan points to be ready in advance because transfers plus a 7-day lead leave little slack.
- Because Flying Blue (359–360 days) and Virgin (331 days) differ in window length, both have been "open" for this window since late 2025; any current absence of Korean Air space reflects Korean Air's partner allocation, not calendar timing.
- The Asiana→Korean Air flight-number cutover on 2026-12-17 (see KQ4) could cause partner (Aeroplan/United) inventory for OZ-coded flights after that date to be re-filed; this is an inference, not a documented pattern.

### Gaps
- No documented 2024–2026 close-in release pattern for Korean Air SKYPASS/partner space, Asiana, Turkish, Qatar, Air Canada-operated or United-operated flights was found (Reddit/FlyerTalk/milemoa were unreachable).
- Time-of-day of Korean Air/Asiana/Flying Blue releases not found.
- United MileagePlus (337 days) and Delta figures were only seen second-hand (AwardWallet quote); the full AwardWallet/AwardFares tables could not be read.

---

## KQ3. Seasonality: is 2026-11-20 → 2026-12-10 low, shoulder or peak for US→Korea?

### Takeaway
Most fare-trend sources classify November and the first half of December (before Christmas) as the low or shoulder season for US→Korea economy fares, with January also low and late December the peak; the main countervailing factors are the US Thanksgiving weekend (Nov 26, 2026 — Hopper projects the 2026 holiday season as the most expensive in a decade) and the Korean Air–Asiana brand cutover on Dec 17, 2026. One Kayak dataset (Virginia origin) contradicts this with Nov/Dec averages far above January. 한 줄 요약: 11/20~12/10은 미국→한국 노선에서 대체로 비수기~어깨철(추수감사절 전후 제외)이라는 근거가 우세하나, 출발지별 편차가 큼.

### Cited Findings
- Kayak (Minnesota→South Korea): "The cheapest month for flights from Minnesota to South Korea is November, when tickets cost $1,211 (return) on average"; Kayak suggests booking 19 weeks before departure — [Kayak MN→KR](https://www.kayak.com/flight-routes/Minnesota-USMN/South-Korea-KR0)
- Kayak (Virginia→South Korea): January cheapest at $1,250 round-trip average, "while December and November average $2,471 and $2,081" — contradicts the Minnesota pattern — [Kayak VA→KR](https://www.kayak.com/flight-routes/Virginia-USVA/South-Korea-KR0)
- Seoul Searching (travel blog, anecdotal): cheapest from California "early-winter (November, mid-December)"; from New York "before the holidays (November, early-December)"; generally "early-December (before Christmas) and mid-January" — [Seoul Searching](https://seoulsearching.net/cheap-travel-south-korea/)
- Travorio (low-quality, mirrored page): "roundtrip fares from major US airports are lowest in January, February, and November"; warns of Chuseok and Lunar New Year peaks — [Travorio](https://travorio.com/ru/guides/cheapest-time-to-fly-to-south-korea)
- Skyscanner Chicago→Seoul (Malaysia site, March 2026 data): cheapest month November; other country versions of the same page say April/May/June/August (internally inconsistent) — [Skyscanner .com.my](https://www.skyscanner.com.my/routes/chia/sela/chicago-to-seoul.html); [Skyscanner .co.za ORD](https://skyscanner.co.za/routes/ord/sela/chicago-o-hare-international-to-seoul.html); [Skyscanner .ie](https://skyscanner.ie/routes/chia/sela/chicago-to-seoul.html)
- Expedia ORD→ICN: "October is typically cheapest", with some of the most affordable fares also in November; round-trip from $881 (undated) — [Expedia ORD–ICN](https://www.expedia.com/es/lp/vuelos/ord/icn/chicago-a-seul)
- Kupi ORD→Seoul monthly table: February and November lowest average one-way fares (page text inconsistently says August); one-way range about $625–$956 by month — [Kupi](https://www.kupi.com/en/flights/from-chicago/to-seoul)
- Hopper 2026 Holiday Travel Index (via syndicated Fox report): "The 2026 holiday travel season is projected to be the most expensive in a decade"; Thanksgiving average domestic round-trip $402, up 31% YoY; driver is "prolonged high jet fuel costs… largely due to the conflict in the Middle East"; international demand led by Tokyo, London, Paris, Bangkok; advice to book before Halloween and avoid the Sunday after Thanksgiving — [Fox 5 Atlanta / Hopper, 2026](https://www.fox5atlanta.com/news/2026-holiday-travel-thanksgiving-christmas-airfares-reach-highest-prices-decade)
- Korean Air's own fare teasers show ORD–ICN economy round trips "from USD 1,816" for late-Aug/early-Sep 2026 and for Oct 7–20, 2026 (fare estimates incl. taxes/surcharges) — [Korean Air US flights page](https://www.koreanair.com/flights/en-us/flights-from-united-states); ANA shows ORD–Seoul round trip Nov 20–Dec 8, 2026 "from $2,635" (collected within 48 h of page view) — [ANA Chicago→South Korea](https://flights.ana.co.jp/en-us/flights-from-chicago-to-south-korea) (see KQ5 for caveats)
- Operational event inside/adjacent to the window: Asiana flights departing on/after Dec 17, 2026 switch to Korean Air flight numbers and Korean Air operation; codeshare transfers Dec 7–9, 2026 — [Asiana notice](https://m.flyasiana.com/C/US/EN/customer/notice/detail?id=CM202608110002530012); reservation re-coding runs Nov 2–Dec 3, 2026 — [Korea Herald Biz](https://biz.heraldcorp.com/article/10872965)
- Seoul events in window (thin): "Guseok Guseok Live" street-performance series runs May 1–Dec 31, 2026 — [HaniSeoul (German listing, anecdotal)](https://www.haniseoul.com/de/events/korea/2720203); November 2026 Seoul concerts listed (e.g., Nov 14, Nov 19) — [Eventworld](https://www.eventworld.com/seoul/concerts/november)
- US Thanksgiving 2026 falls on Thursday Nov 26, 2026 (calendar fact; given in assignment).

### Inferences
- For an outbound around Nov 20–25 and return Dec 5–10, the trip sits in what most datasets call the pre-holiday trough for US–Korea economy pricing; the exception is the Thanksgiving-weekend outbound/return days themselves (Nov 25–29), where Hopper's domestic data suggests demand spikes that typically propagate to connecting US domestic legs rather than to the ORD–ICN nonstop.
- Award availability generally tracks cash demand inversely; a shoulder period plus a 2026 fuel-driven fare spike (Hopper) could mean airlines hold fewer seats for award release close-in, but no source directly measured award availability for this window.
- Korean school winter vacations and year-end business travel would mainly affect Korea-origin outbound demand from mid-/late-December, i.e., after the return date in this window; this is an inference because no Korean school-calendar source could be retrieved.

### Gaps
- No Chicago-specific monthly cash-fare seasonality dataset (Google Flights price insights, Kayak ORD–ICN route page) could be retrieved; state-level Kayak data (MN vs VA) conflict.
- No source on Korean school winter-vacation 2026 start dates (Korean-language search returned unrelated years/regions).
- No award-availability-by-month data for US–Korea; no Korean Air official peak/off-peak (성수기) calendar for 2026 awards could be fetched.
- No confirmed list of large Seoul events (e.g., award shows, conferences) for Nov 20–Dec 10, 2026.

---

## KQ4. ORD→ICN flight picture for winter 2026/27 (nonstops, one-stops, route changes)

### Takeaway
Korean Air operates ORD–ICN daily (7 weekly, KE037/038, 777-300ER per schedule aggregators mid-2026); Asiana's ORD status for NW26 could not be confirmed, but all Asiana flights departing on/after Dec 17, 2026 switch to Korean Air flight numbers/operation (with Korean Air aircraft replacing Asiana A350s on selected US routes), so any OZ-coded ORD service straddling the window will be re-coded. New 2026 US–Korea capacity came at other gateways (United EWR–ICN from Sept 2026, Air Premia IAD–ICN from Apr 24, 2026, Alaska SEA–ICN from Sept 2025), and United adds daily ORD–NRT from Oct 2026 (a new one-stop option). 한 줄 요약: 시카고 직항은 대한항공 매일 1편(77W)이 확인되고, 아시아나 ORD 편은 12/17 이후 KE 편명·KE 운항으로 전환되므로 NW26 ORD 세부 스케줄은 aeroroutes에서 재확인 필요.

### Cited Findings

**Nonstop**
- Korean Air ORD–ICN: "7 weekly departures are currently scheduled on this route"; departure times range 11:15–12:30 depending on schedule period; flight time 14h20–15h; page updated June 29, 2026 — [Directflights ORD–ICN](https://www.directflights.com/ORD-ICN)
- "30 nonstop flights each month from ORD to Incheon Intl" on Korean Air — [Flights.com ORD–ICN](https://www.flights.com/lp/flights/ord/icn/chicago-to-seoul)
- KE037 ICN→ORD "Daily 10:40" departure, 09:30 arrival, Boeing 777-300ER (77W), effective 2026-09-01 — [FlightMapper KE ICN–ORD](https://info.flightmapper.net/route/Korean_Air_KE_ICN_ORD); KE038 ORD→ICN operated by Boeing 777-300 — [Trip.com KE038 status](https://www.trip.com/flights/status-ke038/); Korean Air route page — [Korean Air Chicago→Seoul](https://www.koreanair.com/flights/en-us/flights-from-chicago-to-seoul)
- Asiana ORD–ICN: **no schedule result mentioned Asiana on ORD** (schedule aggregators listed only Korean Air) — [Directflights](https://www.directflights.com/ORD-ICN); [FlightsFrom ORD–ICN (unreadable)](https://www.flightsfrom.com/ORD-ICN); Trip.com's Asiana Seoul–Chicago page lists Korean Air's flight — [Trip.com (unreliable)](https://www.trip.com/hot/asiana-airlines(oz)-flights-from-seoul-to-chicago/)

**Korean Air–Asiana integration (affects the window)**
- Asiana official: flights departing on or after December 17, 2026 "will operate under Korean Air flight numbers and be operated by Korean Air"; codeshare transfers Dec 7–9, 2026; dates subject to change — [Asiana notice CM202608110002530012](https://m.flyasiana.com/C/US/EN/customer/notice/detail?id=CM202608110002530012)
- Re-coding of existing Asiana reservations for flights from Dec 17 runs Nov 2–Dec 3, 2026, with individual KakaoTalk/SMS/email notices; Dec 17 is "the planned launch date of the merged airline" — [Korea Herald Biz](https://biz.heraldcorp.com/article/10872965); [Herald (mobile)](https://mbiz.heraldcorp.com/article/10872965)
- AeroRoutes: OZ-coded flight numbers "from 17DEC26 will gradually transition to KE-coded flight numbers" (filed 03SEP26) — [AeroRoutes KE/OZ NW26 international changes](https://www.aeroroutes.com/eng/260902-keoznw26int)
- AeroRoutes (filed 26AUG26): selected OZ-coded US flights to be operated by Korean Air 777-300ER / 787-10 / A380 from 17DEC26, replacing Asiana A350-900s (ORD not named in snippet) — [AeroRoutes Asiana NW26 US aircraft changes](https://www.aeroroutes.com/eng/260826-oznw26us); JFK: both daily OZ-coded flights on Korean Air 777-300ER from 17DEC26 — [AeroRoutes OZ NW26 JFK](https://www.aeroroutes.com/eng/260909-oznw26jfk); [AeroRoutes KE/OZ NW26 JFK](https://www.aeroroutes.com/eng/260630-keoznw26jfk)
- Seattle: OZ272/271 becomes KE047/048 from 12OCT26–15DEC26; combined service rises from 12 to 14 weekly — [AeroRoutes KE/OZ Seattle](https://www.aeroroutes.com/eng/260917-keoznw26sea)
- Brand retirement: Korean Air to retire the Asiana brand in December; the 69-aircraft fleet may keep Asiana livery for months — [InsideFlyer](https://www.insideflyer.com/posts/korean-air-to-retire-asiana-brand-in-december)
- Asiana moved to ICN Terminal 2 (Korean Air's terminal) on Jan 14, 2026 — [LoyaltyLobby, Nov 5 2025](https://loyaltylobby.com/2025/11/05/asiana-airlines-moves-to-terminal-2-at-seoul-incheon-airport-from-january-14-2026/); [Asiana T2 notice](https://m.flyasiana.com/C/KR/EN/customer/notice/detail?id=CM202509300002526502)
- KFTC remedies: slot/traffic-right cessions on additional routes, remaining 18 routes from early 2026 (Chicago not named) — [ch-aviation](https://www.ch-aviation.com/news/159676-korean-air-asiana-to-cede-slots-on-10-more-routes)
- **UNVERIFIED (anecdotal travel guide):** advice to use Asiana miles before Dec 17, 2026 because the program "closes permanently" after that date — [AirGazette T2 guide](https://www.airgazette.com/incheon-terminal-2-asiana-transit-guide)

**One-stop carriers and 2026 route news**
- United: new daily ORD–Tokyo Narita from October 2026 — [The Traveler](https://www.thetraveler.org/united-adds-sfo-sapporo-and-ord-narita-to-boost-winter-2026-asia-links/); 14 new ORD nonstops in 2026 incl. Narita — [InsideFlyer 2026 routes tag](https://www.insideflyer.com/tag/2026-routes/)
- United: year-round daily Newark–Seoul Incheon from September 2026 — [TravelExtra, notable 2026 routes](https://www.travelextra.ie/?p=103698)
- Air Premia: IAD–ICN nonstop from April 24, 2026 — [Dulles Airport press release](https://www.flydulles.com/news/air-premia-launch-new-iad-service-seoul-south-korea)
- Alaska Airlines: SEA–ICN launched Sept 2025 — [NW Asian Weekly](https://nwasianweekly.com/2025/09/alaska-airlines-launches-nonstop-service-from-seattle-to-seoul)
- Philippine Airlines: ORD–MNL from Nov 9 (3× weekly A350-900) — [Travel and Tour World](https://www.travelandtourworld.com/news/article/gnmttx7i0r5q/)
- American: ORD–NRT daily returns March 27, 2027 (outside window) — [InsideFlyer, American 12 new ORD routes](https://www.insideflyer.com/posts/american-adds-12-new-ord-routes-in-2026/)
- One-stop fare teasers confirming service: Cathay Pacific ORD→Seoul (via HKG) "from USD 1,006" — [Cathay Pacific](https://flights.cathaypacific.com/destinations/en_US/flights-from-chicago-to-seoul); Etihad ORD→Seoul "from USD 2,210" — [Etihad](https://www.etihad.com/nl/flights/vluchten-van-chicago-naar-seoul); ANA ORD→South Korea page — [ANA](https://flights.ana.co.jp/en-us/flights-from-chicago-to-south-korea)
- No announcement of a new ORD–ICN nonstop by any other carrier for 2026 was found (searches returned only existing Korean Air service and other-gateway launches).

### Inferences
- For an outbound Nov 20–25, Asiana-coded ORD flights (if operating) would still be OZ-coded; a return on/after Dec 17 would be KE-coded and possibly on a Korean Air aircraft — but the chosen window ends Dec 10, so the window itself is before the cutover; only the re-coding period (Nov 2–Dec 3) overlaps, which is a change-risk for existing Asiana award bookings.
- Korean Air's daily 777-300ER is the only nonstop confirmed by multiple aggregators; capacity is therefore one widebody per day each way, which explains thin partner award space.
- United's new daily ORD–NRT (Oct 2026) adds a second United-metal Tokyo connection (alongside HND) for Star Alliance one-stops to ICN.

### Gaps
- Asiana ORD–ICN NW26 status (operating or not, frequency, aircraft, Dec 17 KE flight number) could not be confirmed; the relevant AeroRoutes pages exist but were unreadable.
- Winter 2026/27 frequencies/aircraft for the one-stop carriers at ORD (ANA HND/NRT, JAL HND/NRT, EVA TPE, Air Canada YYZ/YVR, Turkish IST, Qatar DOH, Delta hubs, American DFW–ICN) were not retrieved before the search budget was exhausted.
- Korean Air NW26 ORD aircraft confirmation for Nov–Dec specifically (vs the Sept 1, 2026 effective 77W listing) not obtained.

---

## KQ5. Cash-fare benchmarks for ORD–ICN around 2026-11-20 → 2026-12-10 (to value points)

### Takeaway
The only dated benchmark exactly inside the window is ANA's landing-page quote of a Chicago–Seoul round trip Nov 20–Dec 8, 2026 "from $2,635" (one-stop, cabin unstated); Korean Air's own ORD–ICN economy round-trip teasers for fall 2026 are "from USD 1,816"; aggregator floors for the route run roughly $881–$1,006 round-trip (one-stop, undated). For business class, no dated published fare was found — only broker/consolidator ranges ($3,200–$5,000 RT "shoulder season"; one $1,735 discounted quote vs a $4,338 "old price") that are UNVERIFIED. 한 줄 요약: 이코노미 왕복 ≈ $1,800~2,600(직항·해당 기간 근접), 원스탑 바닥가 ≈ $900~1,000; 비즈니스는 검증된 공시 운임 없음(브로커 추정 $3,200~5,000).

### Cited Findings
- ANA: ORD→Seoul round trip, Fri Nov 20, 2026–Tue Dec 8, 2026, "from $2,635"; fares "collected within the last 48 hours" of page capture; cabin not stated in snippet (likely economy — UNVERIFIED) — [ANA Chicago→South Korea](https://flights.ana.co.jp/en-us/flights-from-chicago-to-south-korea)
- Korean Air (official, economy): ORD–ICN round trip "from USD 1,816" for late Aug–early Sep 2026 and again "USD 1,816 for Oct 07, 2026 - Oct 20, 2026"; "estimates that include taxes and fuel surcharges per adult and may change with availability" — [Korean Air US flights page](https://www.koreanair.com/flights/en-us/flights-from-united-states); German-language US page "ab USD 1,816" — [Korean Air de-us Chicago→Seoul](https://www.koreanair.com/flights/de-us/flights-from-chicago-to-seoul)
- Korean Air reverse direction: ICN→ORD round trips mid-November "from KRW 2,141,000" — [Korean Air Seoul→Chicago](https://www.koreanair.com/flights/en/flights-from-seoul-to-chicago)
- Etihad (one-stop via AUH): ORD→Seoul "from USD 2,210"; Seoul→Chicago from KRW 2,934,400 — [Etihad ORD→Seoul](https://www.etihad.com/nl/flights/vluchten-van-chicago-naar-seoul); [Etihad Seoul→Chicago](https://www.etihad.com/nl/flights/vluchten-van-seoul-naar-chicago)
- Cathay Pacific (one-stop via HKG): ORD→Seoul "from USD 1,006"; its business-class filter returned "no fares that match your filter criteria" — [Cathay Pacific](https://flights.cathaypacific.com/destinations/en_US/flights-from-chicago-to-seoul)
- Expedia: ORD→ICN round-trip "from $881" (undated) — [Expedia](https://www.expedia.com/es/lp/vuelos/ord/icn/chicago-a-seul)
- Kupi: ORD→Seoul one-way monthly averages ~$625–$956 (page internally inconsistent) — [Kupi](https://www.kupi.com/en/flights/from-chicago/to-seoul)
- Skyscanner (Ireland site, economy): "cheapest flight found 767 €" Chicago→Seoul — [Skyscanner .ie](https://skyscanner.ie/routes/chia/sela/chicago-to-seoul.html)
- Kayak state averages (not ORD-specific): Minnesota→Korea November avg $1,211 RT; Virginia→Korea November $2,081 / December $2,471 / January $1,250 — [Kayak MN](https://www.kayak.com/flight-routes/Minnesota-USMN/South-Korea-KR0); [Kayak VA](https://www.kayak.com/flight-routes/Virginia-USVA/South-Korea-KR0)
- Business class, **UNVERIFIED broker/consolidator claims**: "$3,200-$5,000 round-trip for business class in shoulder season" (marketing page) — [BusinessClassTravel.us](https://www.businessclasstravel.us/business-class-from/ord-to-seoul); SkyLux lists a discounted ORD→Seoul business fare of $1,735 against an "old price" of $4,338 (undated, cabin/dates unclear) — [SkyLux Travel](https://www.skyluxtravel.com/business-class-flights/?p=16576)
- No Korean Air Prestige (business) "from USD" fare for Chicago appears on Korean Air's own fare pages (only economy for ORD; Prestige teasers shown only for China routes) — [Korean Air US flights page](https://www.koreanair.com/flights/en-us/flights-from-united-states)
- Context for valuation: Hopper projects 2026 holiday fares as the highest in a decade due to jet-fuel costs — [Fox/Hopper 2026](https://www.fox5atlanta.com/news/2026-holiday-travel-thanksgiving-christmas-airfares-reach-highest-prices-decade)

### Inferences
- Reasonable valuation anchors (illustrative, from the sourced numbers): nonstop/near-window economy round trip ≈ $1,800–$2,650 (Korean Air teaser to ANA Nov 20–Dec 8 quote); one-stop economy floor ≈ $900–$1,000 RT (Expedia/Cathay); business round trip ≈ $3,200–$5,000 (broker range, UNVERIFIED — treat as an upper-bound anchor until a dated published fare is captured).
- Cents-per-point formula to apply once the companion researcher supplies award prices: (cash fare − award taxes/fees) ÷ points × 100. Example with the sourced economy anchor $1,816 RT and hypothetical taxes of $150: ($1,816 − $150) ÷ points. (No award prices are asserted here; they belong to the booking-path research.)
- Because every benchmark above is a "from" teaser or aggregator floor rather than a fare for the exact dates/cabin, a Google Flights/ITA capture for ORD–ICN on the actual travel dates (economy and business, nonstop vs one-stop) is still required before computing a defensible cpp.

### Gaps
- No Google Flights price-trend data, no Kayak ORD–ICN route page, and no Korean Air/Asiana/Delta/United published business fare for the window could be retrieved (fetch blocked, search budget exhausted).
- No airline sale announcement (Korean Air/Asiana/United/Delta) covering Nov–Dec 2026 ORD–ICN was found.
- One-way economy and one-way business benchmarks on the nonstop are missing; only round-trip teasers were found.

---

## KQ6. Lap-infant cost basics across programs (percentage-of-fare programs, base-fare determination, typical USD amounts)

### Takeaway
Programs split into three models: percentage of the adult **cash** fare (Flying Blue 10%; KrisFlyer 10% per a comparison chart), percentage of the adult **award miles** (Korean Air SKYPASS 10%, Qatar Privilege Club 10% of Avios, British Airways 10%), and flat charges (Virgin Atlantic 1,000/2,000/5,000 points by cabin, up to 7,000 on partners; Aeroplan 2,500 points or CAD 25 per direction outside Canada/US; United flat $20–$250 by region/cabin since 2021, previously 10% of the cheapest available revenue fare in the ticketed cabin capped at $250). No source gave a typical USD figure for a US–Korea infant award ticket. 한 줄 요약: Flying Blue는 성인 '현금' 운임의 10%라 비즈니스에서 비싸질 수 있고, 대한항공·카타르·BA는 '마일'의 10%, Virgin·Aeroplan·United는 정액.

### Cited Findings
- Korean Air SKYPASS: international lap-infant awards cost 10% of the flight award mileage; cannot be funded with transferable bank points — [TPG, infant/child award tickets](https://thepointsguy.com/news/save-infant-child-award-tickets); [AwardWallet, best airlines for international lap-infant awards](https://awardwallet.com/blog/best-airlines-for-international-lap-infant-award-tickets); [AwardWallet infant-on-lap policies](https://awardwallet.com/airlines/infant-on-lap-award-policies/)
- Flying Blue: infants charged 10% of the adult **cash** fare plus taxes/fees, "which can get expensive in premium cabins" — [TPG](https://thepointsguy.com/news/save-infant-child-award-tickets); [AwardWallet](https://awardwallet.com/blog/best-airlines-for-international-lap-infant-award-tickets)
- Virgin Atlantic Flying Club: flat 1,000 points (Economy), 2,000 (Premium), 5,000 (Upper Class) plus taxes/fees on Virgin metal; on partner flights up to 7,000 points, and carrier surcharges still apply — [Travel on Point(s), lap infant award travel](https://travel-on-points.com/lap-infant-award-travel/); [Prince of Travel, infant flight awards](https://princeoftravel.com/guides/flying-with-kids-infant-flight-awards/)
- Air Canada Aeroplan: infants free within Canada and Canada–US; elsewhere 2,500 points or CAD 25 per direction — [Prince of Travel](https://princeoftravel.com/guides/flying-with-kids-infant-flight-awards/); [10xTravel, award travel with a lap infant](https://10xtravel.com/award-travel-with-a-lap-infant-what-you-need-to-know/)
- United MileagePlus: (older rule) "10% of the revenue (cash) fare for a lap child on an award ticket", with United's rate desk describing the base as "10% of the cheapest available revenue fare in the ticketed cabin" (POTENTIALLY OUTDATED) — [Travel Codex](https://www.travelcodex.com/answers-to-reader-questions-booking-travel-for-infant/); capped at $250 total whether one-way or round trip — [One Mile at a Time](https://onemileatatime.com/?p=275712); [Frequent Miler](https://frequentmiler.com/the-best-programs-for-lap-infant-award-tickets/); (2021 onward) international lap infants "subject to infant fares between $20 and $250 plus taxes depending on area of travel and cabin of service", on an unpublished sliding scale — [View from the Wing](https://viewfromthewing.com/great-improvement-united-now-caps-infant-fares-on-award-tickets-at-250); [TPG, Feb 2021](https://thepointsguy.com/news/united-infant-fees/)
- Qatar Airways Privilege Club: infants cost 10% of the adult Avios; infant awards limited to Qatar-operated flights in Economy — [Australian Frequent Flyer thread (anecdotal)](https://www.australianfrequentflyer.com.au/community/threads/123281/); [AFF Privilege Club overview](https://australianfrequentflyer.com.au/?p=43783)
- British Airways: 10% of flight award mileage (comparison chart, undated) — [Prince of Travel chart](https://princeoftravel.com/guides/flying-with-kids-infant-flight-awards/)
- Singapore KrisFlyer: "10% of adult cash fare" (comparison chart, undated) — [Prince of Travel chart](https://princeoftravel.com/guides/flying-with-kids-infant-flight-awards/)
- Qatar's official page found covers only the unaccompanied-minor service (3,000 Avios per one-way), not lap infants — [Qatar Airways child fares page](https://qatarairways.com/en/family/child-fares-baggage.html)

### Inferences
- For percentage-of-cash programs (Flying Blue, KrisFlyer), the infant charge scales with the adult fare in the booked cabin; applying the KQ5 anchors gives an **illustrative** economy infant cost of roughly 10% × $1,816–$2,635 ≈ $180–$265 round trip, and business of 10% × $3,200–$5,000 ≈ $320–$500 round trip (business anchor UNVERIFIED). The exact base fare (lowest available vs. a specific booking class on the day of ticketing) is program-specific and was only documented for United's old rule.
- For percentage-of-miles programs (SKYPASS, Qatar, BA), the infant cost is independent of cash fares and is cheap in miles but still carries taxes/surcharges; for flat-fee programs (Virgin, Aeroplan, United) the infant cost is predictable and small.

### Gaps
- No source stated how Flying Blue or KrisFlyer determine the "adult cash fare" base for award-ticket infants (published full fare vs. lowest available in cabin vs. the fare of the booking class).
- No documented USD amounts for US–Korea infant tickets on award bookings in 2024–2026 were found (Reddit/FlyerTalk unreachable).
- Current United infant sliding-scale amounts by region/cabin remain unpublished per the sources; no 2025–2026 confirmation of the $20–$250 range was found.
- Korean Air SKYPASS infant policy for Korean Air flights booked through partner programs (Flying Blue/Virgin) not specifically addressed.
