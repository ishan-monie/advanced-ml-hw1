"""Builds report.pdf. The report is built one section at a time; add new sections to SECTIONS below.
Run: python make_report.py"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted,
                                PageBreak, Image, KeepTogether)
from PIL import Image as PILImage

HERE = os.path.dirname(os.path.abspath(__file__))
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontSize=11, leading=15, spaceAfter=8)
TITLE = ParagraphStyle("title", parent=ss["Title"], fontSize=20, spaceAfter=14)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceBefore=14, spaceAfter=8)
CODE = ParagraphStyle("code", parent=ss["Code"], fontSize=9.5, leading=12, backColor=colors.HexColor("#f2f2f2"),
                      borderPadding=6, spaceAfter=10, leftIndent=6)
LINK = ParagraphStyle("link", parent=BODY, textColor=colors.HexColor("#1a55b0"))

GITHUB_USER = "ishan-monie"
REPO_NAME = "advanced-ml-hw1"
REPO_URL = f"https://github.com/{GITHUB_USER}/{REPO_NAME}"


def summary_section(story):
    story.append(Paragraph("Assignment 1 &ndash; Summary", TITLE))

    story.append(Paragraph("GitHub repository", H1))
    t = Table([["GitHub username", GITHUB_USER], ["Repository name", REPO_NAME],
               ["Link to the code", Paragraph(f'<link href="{REPO_URL}">{REPO_URL}</link>', LINK)]],
              colWidths=[1.7 * inch, 4.9 * inch])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 11),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(t)
    story.append(Spacer(1, 8))
    story.append(Paragraph("All of the code used for this assignment is in this repository.", BODY))

    story.append(Paragraph("How to run the code", H1))
    story.append(Paragraph("<b>Step 1.</b> Download the repository (green <i>Code</i> button, then <i>Download ZIP</i>) "
                           "or clone it, and install the Python packages:", BODY))
    story.append(Preformatted("pip install numpy matplotlib", CODE))
    story.append(Paragraph("<b>Step 2.</b> Shuttle problem. Open a terminal in the <font face='Courier'>q1_shuttle</font> folder and run:", BODY))
    story.append(Preformatted("python run_q1.py     # solves the problem 3 ways and saves the 3 plots in figures/\n"
                              "python test_q1.py    # runs the 32 edge-case and correctness checks (PASS/FAIL for each)", CODE))
    story.append(Paragraph("The README in the repository repeats these instructions and explains what each file does. "
                           "Instructions for the policy gradient questions will be added to the same README.", BODY))


CAPTION = ParagraphStyle("caption", parent=BODY, fontSize=9.5, leading=12, alignment=1,
                         textColor=colors.HexColor("#444444"), spaceAfter=12)
FIG_DIR = os.path.join(HERE, "..", "q1_shuttle", "figures")


def figure(story, filename, caption, width=3.2 * inch):
    path = os.path.join(FIG_DIR, filename)
    w, h = PILImage.open(path).size
    story.append(KeepTogether([Image(path, width=width, height=width * h / w), Spacer(1, 3), Paragraph(caption, CAPTION)]))


def shuttle_section(story):
    story.append(PageBreak())
    story.append(Paragraph("Question 2: Value and Policy Iteration (the shuttle problem)", TITLE))
    story.append(Paragraph("Each period we choose whether to dispatch a shuttle (cost 100, takes up to 15 people) or wait "
                           "(every person left waiting costs 2). Between 1 and 5 new people arrive each period. "
                           "We solve this three ways. A lower cost is better in every graph.", BODY))

    story.append(Paragraph("Assumptions", H1))
    story.append(Paragraph("The problem leaves some details open, so these are the choices we made:", BODY))
    assumptions = [
        "<b>What we track.</b> The only thing that matters when deciding is how many people are waiting at the start of a period "
        "(0 to 200). We do not track the time of day or what happened before.",
        "<b>Costs, not rewards.</b> Everything is a cost, so we look for the <i>lowest</i> total cost. The &ldquo;value&rdquo; in the graphs is the "
        "expected total cost, so lower is better.",
        "<b>Order of events in a period.</b> We first decide (wait or dispatch), then pay, then the new people arrive.",
        "<b>Dispatching.</b> The shuttle is always available. It takes up to 15 people, or everybody if fewer than 15 are waiting. "
        "We allow dispatching with an empty station, even though it is wasteful.",
        "<b>The waiting cost.</b> The cost of 2 per person is charged on the people <i>left</i> after the shuttle has gone "
        "(everyone, if we wait).",
        "<b>Full station.</b> The station holds at most 200 people. If more would arrive, the extra people are turned away "
        "(they are lost, they do not wait). We checked that this choice does not change the answer.",
        "<b>Arrivals.</b> Each period 1, 2, 3, 4 or 5 people arrive, each with chance 1/5, independent of the past.",
        "<b>Discounting.</b> A cost one period later counts 0.95 as much. The cost in the first period is not discounted.",
        "<b>The end of the 500 periods (enumeration).</b> After the last period nothing more is paid, so people still "
        "waiting at that point cost nothing extra.",
        "<b>The plan.</b> The best action depends only on the number of people waiting, and it is the same in every "
        "period (this is what value iteration and policy iteration find).",
        "<b>Stopping and ties.</b> Value iteration stops when no value changes by more than 1e-10. Policy iteration starts from "
        "&ldquo;never dispatch&rdquo; and keeps its current action when two actions cost exactly the same.",
    ]
    for a in assumptions:
        story.append(Paragraph(a, ParagraphStyle("li", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=4), bulletText="•"))
    story.append(Spacer(1, 6))

    story.append(Paragraph("(a) Enumeration, 500 periods", H1))
    story.append(Paragraph("We start at the end of the 500 periods, where nothing is left to pay, and work backwards, "
                           "keeping the cheaper of the two actions at every step. The graph shows the best expected total "
                           "cost at time 0 for each number of people waiting.", BODY))
    figure(story, "a_enumeration.png", "Figure 1. Enumeration: best expected cost at time 0 versus people waiting.")

    story.append(Paragraph("(b) Value iteration, no end date", H1))
    story.append(Paragraph("The same step is repeated with no end date until the numbers stop changing (519 rounds). "
                           "The curve is the same as in (a), which is expected: after 500 periods the far future is "
                           "discounted to almost nothing, and the two differ by about 3 in a billion.", BODY))
    figure(story, "b_value_iteration.png", "Figure 2. Value iteration: best expected cost versus people waiting.")
    figure(story, "ab_overlay.png", "Figure 3. Figures 1 and 2 on the same axes. The two curves lie on top of each other.", width=3.2 * inch)

    story.append(Paragraph("(c) Policy iteration", H1))
    story.append(Paragraph("Here we improve a plan instead of the costs. Starting from &ldquo;never dispatch&rdquo;, we measure how good "
                           "the plan is, change it wherever another action is cheaper, and repeat. It settled after 6 rounds. "
                           "The best plan is to wait while fewer than 13 people are waiting and to dispatch at 13 or more.", BODY))
    figure(story, "c_policy_iteration.png", "Figure 4. Policy iteration: best action versus people waiting.")


SMALL = ParagraphStyle("small", parent=BODY, fontSize=9, leading=11, spaceAfter=0)
SMALLB = ParagraphStyle("smallb", parent=SMALL, fontName="Helvetica-Bold")
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=12, spaceAfter=5, keepWithNext=1)
H3 = ParagraphStyle("h3", parent=ss["Heading3"], fontSize=10.5, spaceBefore=9, spaceAfter=4, keepWithNext=1)


def table4(story, headers, rows, widths):
    data = [[Paragraph(h, SMALLB) for h in headers]] + [[Paragraph(c, SMALL) for c in r] for r in rows]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story.append(t)


def checks_table(story, title, rows):
    """rows: (the test and what we expect, the result, the edge case it covers)"""
    story.append(Paragraph(title, H3))
    table4(story, ("Test (and what we expect)", "Result (matched what we expected)", "Edge case"), rows, (3.1, 2.55, 0.85))


def edge_cases_section(story):
    story.append(PageBreak())
    story.append(Paragraph("Edge cases", H1))
    story.append(Paragraph("The unusual situations we considered, and how the code handles each one (P1 to P17).", BODY))
    story.append(Paragraph("Built into the calculation", H3))
    table4(story, ("#", "Unusual situation", "How the code handles it"), [
        ("P1", "Shuttle leaves with fewer than 15 people", "It takes everybody; the number left is never below 0 (<font face='Courier'>max(people - 15, 0)</font>)."),
        ("P2", "Arrivals would push the station past 200", "The station stops at 200 and the extra people are turned away (<font face='Courier'>min(..., 200)</font>)."),
        ("P3", "Chances of the next situation not adding up to 100%", "Each possible arrival number gets exactly 1 / (number of options), so they always add up to 100%."),
        ("P4", "Off-by-one: 0 to 200 is 201 situations, not 200", "Every table is sized 200 + 1."),
        ("P5", "Waiting should not remove anybody", "When we wait, everyone stays; we pay 2 for each."),
        ("P6", "Enumeration needs a starting point", "After the last period the value is 0, then it works backwards from period 499 to 0."),
        ("P7", "Value iteration stopping too early or never", "It stops only when no value changes by more than 1e-10, not after a fixed number of rounds."),
        ("P8", "Policy iteration looping on ties or rounding noise", "It only changes an action if the new one is better by more than 1e-9; otherwise it keeps the current action."),
        ("P9", "Value iteration flip-flopping on exact ties", "Dispatch is chosen only if it is strictly cheaper; a tie means wait."),
        ("P10", "Inaccurate evaluation of a plan", "Policy iteration solves the exact equations instead of approximating."),
        ("P11", "Dispatching an empty station", "Allowed, with no special rule. It is just never the best action."),
    ], (0.45, 2.5, 3.6))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Safety checks added to stop bad inputs and runaway calculations", H3))
    table4(story, ("#", "Unusual situation", "How the code handles it"), [
        ("P12", "Nonsense numbers: no arrivals, negative arrivals, sizes or costs, a station limit of 0, tables that do not match the problem",
         "It stops at the start with a clear message saying what is wrong."),
        ("P13", "A discount of 1 or more (or below 0)", "Rejected. With a discount of 1 the &ldquo;forever&rdquo; problem has no finite answer."),
        ("P14", "Value iteration never settles", "A maximum number of rounds; if it is exceeded the program stops with an error instead of running forever."),
        ("P15", "Policy iteration never settles", "The same maximum for policy iteration."),
        ("P16", "A strange horizon for enumeration (0, negative or a fraction of a period)", "Rejected: it needs a whole number of periods, at least 1."),
        ("P17", "A value turns into NaN or infinity", "The calculation stops with an error, so bad numbers cannot spread quietly."),
    ], (0.45, 3.0, 3.1))
    story.append(Spacer(1, 4))
    story.append(Paragraph("We did not guard against things like typing text instead of a number: that is careless typing, not part of this problem, "
                           "and the assignment fixes all the numbers.", SMALL))

    story.append(PageBreak())
    story.append(Paragraph("Tests", H1))
    story.append(Paragraph("Each test says what we expect to happen, and the result column says what did happen. All 32 tests passed, meaning every result matched the expectation. Run them with <font face='Courier'>python test_q1.py</font>. The last column says which edge case (P1 to P17) a test covers.", BODY))
    checks_table(story, "A. Ordinary situations", [
        ("<b>A1.</b> Empty station (0 people): the best action should be to wait, because a shuttle would cost 100 to carry nobody.",
         "The best action was to wait.", "P11"),
        ("<b>A2.</b> Dispatching with fewer than 15 people (4): the shuttle should take everybody, and we should pay only the 100.",
         "The cost was 100 and the next period had 1 to 5 people.", "P1"),
        ("<b>A3.</b> Does the 200-person limit matter? With a limit of 300 the best actions should not change.",
         "The actions were the same up to 150 people, and the costs differed by only 9e-11.", "P2"),
        ("<b>A4.</b> The chances of the next situation should add up to 100% everywhere.",
         "The largest error was 0, and a deliberately damaged table was caught by the same check.", "P3"),
        ("<b>A5.</b> A bigger shuttle should wait for more people before being sent.",
         "Shuttles of size 5, 15 and 30 dispatched at 5, 13 and 17 people.", "&ndash;"),
    ])
    checks_table(story, "B. Edges of the station", [
        ("<b>B1.</b> Exactly 15 people should empty the station, while 16 should leave one behind.",
         "The costs were 100 and 102.", "P1"),
        ("<b>B2.</b> Full station (200): we should dispatch, and waiting should not push it above 200.",
         "The best action was to dispatch, and waiting stayed at 200.", "P2"),
        ("<b>B3.</b> Nearly full (199 and 198): at 199 every arrival should be turned away, and at 198 only one should fit.",
         "From 199 the station always went to 200. From 198 it went to 199 with 20% chance and to 200 with 80%.", "P2"),
        ("<b>B4.</b> Off-by-one: there should be 201 situations (0 to 200), not 200.",
         "All tables are 201 wide and the graphs have 201 points.", "P4"),
    ])
    checks_table(story, "C. Arrivals", [
        ("<b>C1.</b> Nobody ever arrives and 10 are waiting (worked by hand): waiting forever costs 20 / (1 - 0.95) = 400 and dispatching once costs 100, "
         "so the cheapest should be 100.",
         "The program gave 100.000.", "&ndash;"),
        ("<b>C2.</b> Far more arrivals than the shuttle can carry (16 to 20 per period): the queue never empties, so each person left costs "
         "2 / (1 - 0.95) = 40 forever. A dispatch (100) should pay off from 100 / 40, so from 3 people. "
         "(Our first guess was 15; working it through showed the guess was wrong, not the program.)",
         "It dispatched for every number from 3 to 152.", "&ndash;"),
        ("<b>C2b.</b> Why does it stop at 152? Near the limit the queue hits 200 anyway, so a dispatch saves little, "
         "and the stopping point should move if the limit moves.",
         "The last dispatch was at 152 with a limit of 200, and at 252 with a limit of 300, so it is an effect of the cap.", "P2"),
    ])
    checks_table(story, "D. Extreme prices", [
        ("<b>D1a.</b> Free shuttle: we should dispatch whenever anyone is waiting.",
         "It dispatched in all 200 situations with someone waiting.", "&ndash;"),
        ("<b>D1b.</b> Free waiting: we should never dispatch, and the total cost should be 0.",
         "It never dispatched, and the cost was 0.", "&ndash;"),
        ("<b>D2.</b> Enormous shuttle cost (1,000,000): we should never dispatch, even with a full station.",
         "It never dispatched.", "&ndash;"),
    ])
    checks_table(story, "E. Time and discounting", [
        ("<b>E1.</b> Very short horizon (worked by hand): with 1 period left the cost should be 2 times the number of people, "
         "and with 2 periods left 3.9 times the people plus 5.7.",
         "Both matched the program exactly.", "P6"),
        ("<b>E2.</b> Is 500 periods long enough? As the horizon grows from 100 to 200 to 500 periods, the answer should settle down.",
         "The difference from the no-end-date answer was 4.0, then 0.024, then 0.000000003.", "P6"),
    ])
    checks_table(story, "F. Checks on each method", [
        ("<b>F1.</b> Enumeration starting point: nothing should be left to pay after the last period, and more periods left should never cost less.",
         "The value after period 500 is 0 everywhere, and costs only grew with more periods.", "P6"),
        ("<b>F2.</b> Value iteration should reach the same answer from different starting guesses.",
         "Starting guesses of all 0, all 1000 and random numbers agreed to 4e-9.", "P7"),
        ("<b>F3.</b> Policy iteration should improve the plan every round, and when all actions cost the same it should not loop forever.",
         "It took 6 rounds and the cost never went up. A problem where every action ties stopped after 1 round.", "P8, P9"),
    ])
    checks_table(story, "G. Overall checks", [
        ("<b>G1.</b> All three methods should agree on the costs and on the plan.",
         "The biggest gap was 3e-9, and the plan was identical.", "all"),
        ("<b>G2.</b> Policy iteration started from a bad plan (&ldquo;always dispatch&rdquo;) should still find the same best plan.",
         "It did.", "P8"),
        ("<b>G3.</b> Simulating the station 3000 times should give an average cost that matches the calculated cost.",
         "The simulation gave 575.8 (plus or minus 1.7) against a calculated 576.7.", "&ndash;"),
    ])
    checks_table(story, "H. The safety checks themselves", [
        ("<b>H1.</b> Nonsense numbers (no arrivals, negative arrivals, sizes or costs, a limit of 0) should each be rejected with a clear message.",
         "All 6 were rejected.", "P12"),
        ("<b>H1b.</b> Tables of the wrong size, or whose chances do not add to 100%, should be rejected.",
         "Both were stopped.", "P12"),
        ("<b>H2.</b> A discount of 1.0, 1.5 or -0.1 should be rejected, while a discount of 0.0 should be allowed.",
         "The first three were rejected and 0.0 was accepted.", "P13"),
        ("<b>H3.</b> Value iteration given a target it can never meet should stop with an error after the maximum number of rounds.",
         "It stopped with an error (50 rounds were allowed).", "P14"),
        ("<b>H4.</b> Policy iteration allowed only 1 round (it needs 6) should stop with an error.",
         "It stopped with an error.", "P15"),
        ("<b>H5.</b> Enumeration with 0, -3 or 2.5 periods should be rejected.",
         "All three were rejected.", "P16"),
        ("<b>H6.</b> A NaN or infinite cost should make all three methods stop with an error.",
         "All three methods stopped. (An infinite cost on an action nobody uses is rightly not an error.)", "P17"),
        ("<b>H7.</b> Waiting with 10 people: we should pay 20, and next period should have 11 to 15 people.",
         "The cost was 20, with a 20% chance of each of 11 to 15 people.", "P5"),
        ("<b>H8.</b> Exact equation solving should agree with a slow repeat-until-settled calculation.",
         "The largest gap was 9e-13.", "P10"),
    ])


def three_months_section(story):
    block = []
    story_real, story = story, block
    story.append(Paragraph("With 3 more months", H1))
    story.append(Paragraph("<b>Make the model more realistic.</b> We assumed 1 to 5 people arrive with equal chance. With real arrival data we would "
                           "check whether &ldquo;dispatch at 13&rdquo; still holds, and add rush hours, travel time, several shuttles and people who give up "
                           "waiting. We would also see how far the answer moves if the costs of 100 and 2 are really 50% to 150% of what we assumed, and ask "
                           "someone who runs a station whether our assumptions (such as when the waiting cost is charged) match reality.", BODY))
    story.append(Paragraph("<b>Test the solution harder.</b> We would generate hundreds of random shuttle problems and check that all three methods agree "
                           "on every one, compare against a brute-force search over every plan on small versions, and simulate rival rules (&ldquo;dispatch "
                           "at 10&rdquo;, &ldquo;at 15&rdquo;, &ldquo;at 20&rdquo;) for a long time to show with confidence ranges that 13 really is cheapest.", BODY))
    story.append(Paragraph("<b>Check it a different way.</b> We would let a learning method, like the policy gradient methods in the next questions, "
                           "learn from simulated stations without being told the rules, and see whether it also finds about 13. We would also check that "
                           "the program stays fast and accurate on a much bigger station.", BODY))
    story_real.append(KeepTogether(block))


PG_FIG_DIR = os.path.join(HERE, "..", "q3_policy_gradient", "figures")


def figure_from(story, folder, filename, caption, width=3.7 * inch):
    path = os.path.join(folder, filename)
    w, h = PILImage.open(path).size
    story.append(KeepTogether([Image(path, width=width, height=width * h / w), Spacer(1, 3), Paragraph(caption, CAPTION)]))


def policy_gradient_section(story):
    story.append(PageBreak())
    story.append(Paragraph("Question 3: Policy Gradient, Part 1", TITLE))
    story.append(Paragraph("We teach a small neural network to play a game by trial and error. It outputs a probability for each action and plays "
                           "a whole game choosing actions at random from those probabilities. Afterwards, actions that were followed by lots of "
                           "reward are made more likely, and actions followed by little reward are made less likely. This is the basic "
                           "version, with no baseline (the next part adds one).", BODY))

    story.append(Paragraph("Assumptions", H1))
    story.append(Paragraph("These are the choices we made for part (a), CartPole-v1:", BODY))
    assumptions = [
        "<b>One game is one episode.</b> CartPole ends when the pole tips too far, the cart leaves the track, or after 500 steps. "
        "Every step the pole stays up gives a reward of 1, so the best possible score is 500.",
        "<b>The network.</b> It sees 4 numbers (cart position, cart speed, pole angle, pole speed), has one hidden layer of 128 units, "
        "and outputs the probability of pushing left or right.",
        "<b>Actions are random.</b> Both during training and during the final 500 games, each push is chosen at random using the network&rsquo;s "
        "probabilities, rather than always taking the most likely one.",
        "<b>Credit for each push.</b> Each push is credited with the reward from that step onwards, discounted by 0.95 per step "
        "(as the assignment says), not with the whole game&rsquo;s total.",
        "<b>Learning.</b> One update after every game, using Adam with a learning rate of 0.01 (as the assignment says). "
        "There is no baseline and the credits are not rescaled, because this is the basic version.",
        "<b>Training length and luck.</b> We train for 1500 games. Results depend on the random starting point, so the main model "
        "uses seed 0, chosen in advance, and two more seeds are shown to see how much luck matters.",
        "<b>Final test.</b> After training the network stops learning. We play 500 new games, with a different random seed, and plot the scores.",
        "<b>Moving average.</b> The average of the last 100 games (fewer at the very start).",
        "<b>The 500-step limit</b> is treated like any other end of the game.",
        "<b>Bad numbers.</b> If an update ever produced NaN or infinity, it would be skipped instead of damaging the network. "
        "This never happened in our runs.",
    ]
    for a in assumptions:
        story.append(Paragraph(a, ParagraphStyle("li3", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=4), bulletText="•"))

    story.append(PageBreak())
    story.append(Paragraph("(a) CartPole-v1", H1))
    story.append(Paragraph("We tried learning rates of 0.01, 0.003 and 0.001 on three seeds each. The averages of the last 100 training games were "
                           "285, 249 and 234, but they varied a lot between seeds (for example 500 against 163 at the same rate), so no rate "
                           "was clearly better and we kept the assignment&rsquo;s 0.01.", BODY))
    figure_from(story, PG_FIG_DIR, "cartpole_training.png",
                "Figure 5. Reward of every training game (grey) and the moving average of the last 100 games (red).", width=4.1 * inch)
    story.append(Paragraph("The network clearly learns: early games last about 20 steps and later ones often reach 200 or more. But it is "
                           "unsteady. The average rises, falls back, and rises again, and the main model finished at an average of about 164 "
                           "over its last 100 training games. This noise is typical of the basic method and is what a baseline is meant to reduce.", BODY))
    figure_from(story, PG_FIG_DIR, "cartpole_histogram.png",
                "Figure 6. Scores of 500 games played by the trained main model (mean 155.4, standard deviation 12.8).", width=3.6 * inch)
    figure_from(story, PG_FIG_DIR, "cartpole_seeds.png",
                "Figure 7. The same training with three different random seeds. Seed 1 reaches the 500 limit and stays there; seeds 0 and 2 do not.", width=4.1 * inch)
    story.append(Paragraph("On the 500 test games, the main model (seed 0) scored a mean of 155.4 with a standard deviation of 12.8. "
                           "The other seeds scored a mean of 500.0 (seed 1, with no variation) and 277.5 (seed 2). Where the model ends up "
                           "depends a lot on the seed, so a single run should not be over-interpreted.", BODY))


def cartpole_edge_cases_section(story):
    story.append(PageBreak())
    story.append(Paragraph("CartPole: Edge cases", H1))
    story.append(Paragraph("The unusual situations we considered for part (a), and how the code handles each one (R1 to R9).", BODY))
    table4(story, ("#", "Unusual situation", "How the code handles it"), [
        ("R1", "A reward crediting pushes that came before it", "Credits are worked out backwards with a running total, so each push only gets the reward from its own step onwards."),
        ("R2", "Bad settings: fewer than 1 game, a learning rate of 0 or below, a discount below 0 or of 1 or more",
         "Rejected at the start with a clear message. With a discount of 1 or more the credits could grow without limit."),
        ("R3", "A NaN or infinite number in the learning step", "That update is skipped and counted, so the network cannot be damaged. A warning is printed if it ever happens (it did not in our runs)."),
        ("R4", "An empty or very short game", "An empty game is skipped. A one-step game still gives a valid update."),
        ("R5", "Taking the log of a probability that is 0, or huge numbers", "Probabilities are built from the raw scores, so they are never exactly 0 and stay finite even for extreme networks or inputs."),
        ("R6", "How a game ends", "The pole falling and the 500-step limit both simply end the game. We use CartPole-v1, so a score can never be above 500."),
        ("R7", "Test games changing the network by accident", "Test games switch off learning information, and the network is not updated during them."),
        ("R8", "Results that cannot be repeated", "Random seeds are fixed, and the 500 test games use a different seed from training."),
        ("R9", "The moving average at the start, before 100 games exist", "It averages over however many games exist so far, instead of failing or counting empty games as 0."),
    ], (0.45, 2.45, 3.6))

    story.append(PageBreak())
    story.append(Paragraph("CartPole: Tests", H1))
    story.append(Paragraph("Each test says what we expect to happen, and the result column says what did happen. All 24 tests passed, meaning "
                           "every result matched the expectation. Run them with <font face='Courier'>python test_cartpole.py</font>. "
                           "The last column says which edge case (R1 to R9) a test covers.", BODY))
    checks_table(story, "A. Credits and game rules", [
        ("<b>A1.</b> Worked by hand: rewards 1, 1, 1 with discount 0.5 should give credits 1.75, 1.5 and 1.0, and a reward should never credit earlier pushes.",
         "The credits were 1.75, 1.5 and 1.0, and a lone reward of 5 credited only its own step.", "R1"),
        ("<b>A2.</b> A one-step game (the pole falls straight away) should still give a valid update.",
         "The update worked and the network changed.", "R4"),
        ("<b>A3.</b> A discount of 0 should give each push only its own reward, 0.99 should work, and 1.0 or -0.1 should be rejected.",
         "Discounts of 0 and 0.99 worked; 1.0 and -0.1 were rejected.", "R2"),
        ("<b>A4.</b> In the longest game (500 steps) no credit should exceed 1 / (1 - 0.95) = 20.",
         "The largest credit was 20.000.", "&ndash;"),
        ("<b>A5.</b> We should be using CartPole-v1 with a 500-step limit (not v0, which stops at 200).",
         "It is CartPole-v1, with a limit of 500.", "R6"),
        ("<b>A6.</b> Every step should give a reward of 1, so a game&rsquo;s score equals its length.",
         "A sample game of 36 steps scored 36.", "&ndash;"),
    ])
    checks_table(story, "B. The network", [
        ("<b>B1.</b> Before any training, both actions should be about equally likely.",
         "The average chance of pushing left was 0.527.", "&ndash;"),
        ("<b>B2.</b> The two probabilities should add up to 100% for every state, with no NaN.",
         "The largest error was about 1e-7.", "R5"),
        ("<b>B3.</b> An extreme network (weights multiplied by 10,000) and extreme inputs (millions) should still give finite log-probabilities.",
         "They did.", "R5"),
    ])
    checks_table(story, "C. The training loop", [
        ("<b>C1.</b> A NaN reward should make the update skip, leaving the network untouched.",
         "The update was skipped and the weights were identical.", "R3"),
        ("<b>C2.</b> An empty game should be skipped instead of crashing.", "It was skipped.", "R4"),
        ("<b>C3.</b> Bad settings (0 games, a learning rate of 0 or below) should be rejected with a message.",
         "All three were rejected.", "R2"),
        ("<b>C4.</b> The same seed should give an identical training curve, and a different seed a different one.",
         "Two runs with seed 3 were identical, and seed 4 differed.", "R8"),
        ("<b>C5.</b> Test games should never change the network.",
         "The weights were identical before and after, and no learning information was kept.", "R7"),
        ("<b>C6.</b> A made-up problem with a known answer (one state where action 1 always pays 1 and action 0 pays 0): "
         "our learning signal should match the exact gradient of the expected reward.",
         "The direction matched (1.0000) and the size was within 0.9%.", "&ndash;"),
        ("<b>C6b.</b> The previous check should really be able to fail: a learner with the sign flipped, or one that ignores the reward, should be caught.",
         "Both were caught (a direction match of -1.00 instead of +1.00).", "&ndash;"),
        ("<b>C7.</b> In the same made-up problem, training should make the better action much more likely.",
         "Its chance went from 0.47 to 1.00.", "&ndash;"),
    ])
    checks_table(story, "D. Does it behave as predicted?", [
        ("<b>D1.</b> The trained network should be far better than an untrained one.",
         "155.4 against 22.0 on average (a statistical test gave t = 176).", "&ndash;"),
        ("<b>D2.</b> With a tiny learning rate (1e-6) it should hardly learn and stay near random play (about 22).",
         "The average of the last 100 games was 20.2.", "&ndash;"),
        ("<b>D3.</b> Taking away the randomness (always the most likely action) should do about as well or better.",
         "155.9 against 155.4.", "&ndash;"),
        ("<b>D4.</b> The last 100 training games should score much higher than the first 100.",
         "The average was 31.4 at the start and 163.5 at the end.", "&ndash;"),
    ])
    checks_table(story, "E. The 500 test games and the graphs", [
        ("<b>E1.</b> There should be exactly 500 test games, each scoring between 1 and 500.",
         "500 games, with scores from 128 to 192, a mean of 155.4 and a standard deviation of 12.8.", "R6"),
        ("<b>E2.</b> The moving average, worked by hand: [1, 2, 3, 4] with a window of 2 should give 1, 1.5, 2.5, 3.5. With a window of 100 the first value "
         "should be the first game, game 100 the mean of games 1 to 100, and game 101 should drop game 1.",
         "All of these matched.", "R9"),
        ("<b>E3.</b> Re-running should reproduce exactly the numbers behind the graphs in this report.",
         "The training curve and the 500 test scores were identical.", "R8"),
    ])


SECTIONS = [summary_section, shuttle_section, edge_cases_section, three_months_section, policy_gradient_section, cartpole_edge_cases_section]

story = []
for add in SECTIONS:
    add(story)
SimpleDocTemplate(os.path.join(HERE, "report.pdf"), pagesize=letter, leftMargin=inch, rightMargin=inch,
                  topMargin=0.9 * inch, bottomMargin=0.9 * inch, title="Advanced ML Assignment 1").build(story)
print("wrote report.pdf")
