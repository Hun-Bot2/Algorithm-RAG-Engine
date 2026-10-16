# Curriculum sources and access boundary v1

The reviewed seed has 90 problem links: 60 LeetCode, 20 AtCoder and 10
Codeforces. Titles, short concept rationales and planning estimates in
`curriculum/annotations.v1.json` are independently authored metadata. No
problem statement, test case, editorial, solution or historical vector is
redistributed here. A canonical link and our own annotation do not grant rights
to copy platform content or imply an official partnership.

## Official statement-access checks on 2026-10-16

Sol checked the finite, curated list of 90 retained candidates on official
platform pages: all 60 LeetCode, all 20 AtCoder and all 10 Codeforces. Eighty
seven statements were visible at their direct task pages; three AtCoder direct
routes were inaccessible to the web inspector, but their statements and task
titles were visible on their official contest printable pages. The individual
`access_provenance` field records the inspected route for every problem.
`access_state=free` means the statement was publicly visible without a paid
subscription gate during this review. It does not establish anonymous or
account-based submission, future availability, availability from every region,
or the right to redistribute platform content.

All 60 LeetCode pages showed a public statement and a native Easy, Medium or
Hard label. Examples include [Two Sum](https://leetcode.com/problems/two-sum/),
[Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/) and
[Number of Islands](https://leetcode.com/problems/number-of-islands/). The full
set of 60 canonical links and individually observed native labels is in
`curriculum/seed.v1.json` and its annotation registry. The separate
`difficulty_band` is editorial planning guidance, not measured learner time.

All 10 Codeforces pages showed a public statement and native rating 800;
examples include [4A](https://codeforces.com/problemset/problem/4/A),
[158A](https://codeforces.com/problemset/problem/158/A) and
[791A](https://codeforces.com/problemset/problem/791/A). All 20 AtCoder
statements were visible, 17 at direct task pages such as
[ABC086 A](https://atcoder.jp/contests/abc086/tasks/abc086_a) and
[ABC104 C](https://atcoder.jp/contests/abc104/tasks/abc104_c). AtCoder pages
show contest points, but points were not converted into a cross-contest or
cross-platform difficulty rating; native difficulty stays `unranked`.

The web inspector could not open the direct task pages for
[ABC049 C](https://atcoder.jp/contests/abc049/tasks/abc049_c),
[ABC086 C](https://atcoder.jp/contests/abc086/tasks/abc086_c), or
[ABC095 C](https://atcoder.jp/contests/abc095/tasks/abc095_c). Their official
printable contest pages displayed the English task identity and statement:
[ABC049 print](https://atcoder.jp/contests/abc049/tasks_print),
[ABC086 print](https://atcoder.jp/contests/abc086/tasks_print?lang=en), and
[ABC095 print](https://atcoder.jp/contests/abc095/tasks_print?lang=en).
These print pages establish public statement access, while direct-link behavior
remains unresolved in the inspector. A browser check from the intended user
location is still useful before relying on those canonical links in missions.
A tool-level fetch failure was not treated as evidence of a paywall.

Three draft LeetCode candidates were removed after their official pages showed
the Premium subscription gate: [Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/),
[Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/), and
[Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/). They were
replaced with public [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/),
[Redundant Connection](https://leetcode.com/problems/redundant-connection/),
and [Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/).

## Rights and remaining verification

[LeetCode's terms](https://leetcode.com/terms/) reserve rights to problem
content and prohibit crawling or scraping. [AtCoder's terms](https://atcoder.jp/tos)
reserve rights to site text and other materials. [Codeforces terms](https://codeforces.com/terms)
apply to its site. This review made finite read-only checks of named pages and
stores only our own annotations plus links. It does not claim permission to
republish statements, clear historical dataset rights, or verify any judge
submission route. The algorithm review and judgment adjudication are recorded
separately in `curriculum-review.md`.
