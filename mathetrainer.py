import random
import time
import pygame

# ----------------------------
# SETTINGS
# ----------------------------
FPS = 60
MAX_LEVEL = 5

CORRECT_PAUSE_SECONDS = 0.6

# doubled per your request
CORRECTS_TO_LEVEL_UP = 10
WRONGS_TO_LEVEL_DOWN = 3

TOTAL_TARGET_CORRECT = MAX_LEVEL * CORRECTS_TO_LEVEL_UP  # 50

# Keep results under 100 for early levels (1..4)
EARLY_LEVEL_CAP = 99


# ----------------------------
# PROBLEM GENERATOR
# ----------------------------
def generate_problem(level: int):
    """
    Returns: (a, op, b, answer)
    op is '+' or '-'
    """

    def pick_op():
        return random.choice(['+', '-'])

    if level == 1:
        # 0..10 results, no negatives
        while True:
            a = random.randint(0, 10)
            b = random.randint(0, 10)
            op = pick_op()
            ans = a + b if op == '+' else a - b
            if 0 <= ans <= 10:
                return a, op, b, ans

    if level == 2:
        # 0..20 results, no negatives
        while True:
            a = random.randint(0, 20)
            b = random.randint(0, 20)
            op = pick_op()
            ans = a + b if op == '+' else a - b
            if 0 <= ans <= 20:
                return a, op, b, ans

    if level == 3:
        # tens only, no negatives; keep < 100
        tens = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        while True:
            a = random.choice(tens)
            b = random.choice(tens)
            op = pick_op()
            ans = a + b if op == '+' else a - b
            if 0 <= ans <= EARLY_LEVEL_CAP:
                return a, op, b, ans

    if level == 4:
        # two-digit "easy" (no carry/borrow), no negatives, keep < 100
        while True:
            op = pick_op()
            if op == '+':
                a10 = random.randint(1, 9)
                b10 = random.randint(1, 9)
                a1 = random.randint(0, 9)
                b1 = random.randint(0, 9)
                if a1 + b1 <= 9:
                    a = a10 * 10 + a1
                    b = b10 * 10 + b1
                    ans = a + b
                    if ans <= EARLY_LEVEL_CAP:
                        return a, op, b, ans
            else:
                a10 = random.randint(1, 9)
                b10 = random.randint(1, 9)
                a1 = random.randint(0, 9)
                b1 = random.randint(0, 9)
                if a10 >= b10 and a1 >= b1:
                    a = a10 * 10 + a1
                    b = b10 * 10 + b1
                    ans = a - b
                    if ans >= 0:
                        return a, op, b, ans

    # Level 5: full two-digit add/sub, subtraction non-negative
    while True:
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        op = pick_op()
        ans = a + b if op == '+' else a - b
        if op == '+' or ans >= 0:
            return a, op, b, ans


# ----------------------------
# TREE DRAWING
# ----------------------------
def draw_leaf(surface, x, y, size, color=(70, 200, 120)):
    # simple ellipse leaf
    rect = pygame.Rect(0, 0, int(size * 1.4), int(size * 0.9))
    rect.center = (x, y)
    pygame.draw.ellipse(surface, color, rect)
    pygame.draw.ellipse(surface, (50, 160, 90), rect, 2)


def draw_tree_panel(surface, panel_rect, correct_by_level, current_level, completed_levels, completed):
    """
    Tree with trunk + 5 branches (one per level).
    Each correct adds a leaf on its branch.
    When level completed: fruit at branch end.
    """
    # background
    pygame.draw.rect(surface, (14, 14, 20), panel_rect)

    cx = panel_rect.centerx
    bottom = panel_rect.bottom - int(panel_rect.height * 0.08)
    top = panel_rect.top + int(panel_rect.height * 0.12)

    # trunk
    trunk_color = (140, 95, 60)
    pygame.draw.line(surface, trunk_color, (cx, bottom), (cx, top), 18)

    # branch anchor points along trunk
    # levels 1..5: from lower to upper
    branch_ys = []
    for i in range(MAX_LEVEL):
        t = (i + 1) / (MAX_LEVEL + 1)  # 1/6..5/6
        y = bottom - int(t * (bottom - top))
        branch_ys.append(y)

    # draw branches and leaves
    for idx in range(MAX_LEVEL):
        lvl = idx + 1
        y = branch_ys[idx]

        # alternate sides
        side = 1 if idx % 2 == 0 else -1
        branch_len = int(panel_rect.width * 0.33)
        end_x = cx + side * branch_len
        end_y = y - int(panel_rect.height * 0.03)

        # branch line
        pygame.draw.line(surface, trunk_color, (cx, y), (end_x, end_y), 10)

        # leaves: up to CORRECTS_TO_LEVEL_UP per level
        leaves = max(0, min(correct_by_level[idx], CORRECTS_TO_LEVEL_UP))
        if leaves > 0:
            for k in range(leaves):
                # place leaves spaced along branch
                s = (k + 1) / (CORRECTS_TO_LEVEL_UP + 1)
                lx = int(cx + (end_x - cx) * s)
                ly = int(y + (end_y - y) * s) - (6 if k % 2 == 0 else -6)
                size = max(10, int(panel_rect.width * 0.05))
                draw_leaf(surface, lx, ly, size=size)

        # fruit if completed
        if lvl in completed_levels:
            fruit_r = max(10, int(panel_rect.width * 0.035))
            pygame.draw.circle(surface, (220, 90, 90), (end_x, end_y), fruit_r)
            pygame.draw.circle(surface, (180, 60, 60), (end_x, end_y), fruit_r, 3)

    # text
    font_title = pygame.font.SysFont(None, max(18, int(panel_rect.width * 0.13)))
    font_small = pygame.font.SysFont(None, max(16, int(panel_rect.width * 0.075)))

    title = font_title.render("TREE", True, (200, 200, 210))
    surface.blit(title, (panel_rect.left + 16, panel_rect.top + 12))

    total_correct = sum(correct_by_level)
    ptxt = font_small.render(f"Progress: {min(total_correct, TOTAL_TARGET_CORRECT)}/{TOTAL_TARGET_CORRECT}", True, (160, 160, 175))
    surface.blit(ptxt, (panel_rect.left + 16, panel_rect.top + 46))

    ltxt = font_small.render(f"Level: {current_level}  (target {CORRECTS_TO_LEVEL_UP})", True, (160, 160, 175))
    surface.blit(ltxt, (panel_rect.left + 16, panel_rect.top + 70))

    if completed:
        done = font_title.render("COMPLETED", True, (80, 220, 120))
        drect = done.get_rect(center=(panel_rect.centerx, panel_rect.bottom - 46))
        surface.blit(done, drect)
        hint = font_small.render("Press ESC to exit", True, (160, 160, 175))
        hrect = hint.get_rect(center=(panel_rect.centerx, panel_rect.bottom - 20))
        surface.blit(hint, hrect)


# ----------------------------
# MAIN
# ----------------------------
def main():
    pygame.init()
    pygame.display.set_caption("Mathetrainer - Tree")

    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    w, h = screen.get_size()
    clock = pygame.time.Clock()

    # Layout: left tasks, right tree
    panel_w = int(w * 0.32)
    task_rect = pygame.Rect(0, 0, w - panel_w, h)
    tree_rect = pygame.Rect(w - panel_w, 0, panel_w, h)

    base = max(24, min(task_rect.width, h) // 10)
    font_task = pygame.font.SysFont(None, int(base * 1.5))
    font_input = pygame.font.SysFont(None, int(base * 1.1))
    font_hint = pygame.font.SysFont(None, int(base * 0.45))

    level = 1
    correct_in_level = 0
    wrong_in_level = 0

    # track progress per level (index 0..4)
    correct_by_level = [0] * MAX_LEVEL
    completed_levels = set()  # levels that reached CORRECTS_TO_LEVEL_UP
    completed = False

    a, op, b, answer = generate_problem(level)
    user_text = ""

    feedback = None  # None / "correct" / "wrong"
    feedback_since = 0.0
    solved_this_problem = False

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

                # when completed: stop interaction except ESC
                if completed:
                    continue

                # ignore input during correct pause
                if feedback == "correct":
                    continue

                if event.key == pygame.K_RETURN:
                    if user_text.strip() == "":
                        continue

                    try:
                        val = int(user_text)
                    except ValueError:
                        feedback = "wrong"
                        feedback_since = time.time()
                        continue

                    if val == answer:
                        feedback = "correct"
                        feedback_since = time.time()

                        if not solved_this_problem:
                            solved_this_problem = True
                            correct_in_level += 1
                            correct_by_level[level - 1] += 1

                        # Level completion / move
                        if correct_in_level >= CORRECTS_TO_LEVEL_UP:
                            completed_levels.add(level)

                            # reset for next level
                            correct_in_level = 0
                            wrong_in_level = 0

                            if level < MAX_LEVEL:
                                level += 1
                            else:
                                # finished last level
                                if sum(correct_by_level) >= TOTAL_TARGET_CORRECT:
                                    completed = True

                    else:
                        feedback = "wrong"
                        feedback_since = time.time()
                        wrong_in_level += 1

                        if wrong_in_level >= WRONGS_TO_LEVEL_DOWN and level > 1:
                            level -= 1
                            correct_in_level = 0
                            wrong_in_level = 0

                elif event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                else:
                    # digits only
                    ch = event.unicode
                    if ch.isdigit():
                        if len(user_text) < 4:
                            user_text += ch

        # next problem after correct pause
        if not completed and feedback == "correct":
            if time.time() - feedback_since >= CORRECT_PAUSE_SECONDS:
                a, op, b, answer = generate_problem(level)
                user_text = ""
                feedback = None
                solved_this_problem = False

        # ----------------------------
        # DRAW
        # ----------------------------
        screen.fill((10, 10, 14))

        # task area
        pygame.draw.rect(screen, (10, 10, 14), task_rect)

        # problem text
        task_text = f"{a}  {op}  {b}  ="
        surf_task = font_task.render(task_text, True, (240, 240, 240))
        rect_task = surf_task.get_rect(center=(task_rect.centerx, int(h * 0.40)))
        screen.blit(surf_task, rect_task)

        # input
        if completed:
            input_color = (200, 200, 210)
        else:
            if feedback == "correct":
                input_color = (80, 220, 120)
            elif feedback == "wrong":
                input_color = (240, 90, 90)
            else:
                input_color = (230, 230, 230)

        shown_input = user_text if user_text != "" else " "
        surf_in = font_input.render(shown_input, True, input_color)
        rect_in = surf_in.get_rect(center=(task_rect.centerx, int(h * 0.55)))
        screen.blit(surf_in, rect_in)

        # hint (ASCII only)
        hint = "ESC: exit   ENTER: check   BACKSPACE: delete"
        surf_hint = font_hint.render(hint, True, (160, 160, 170))
        screen.blit(surf_hint, (task_rect.left + 24, int(h * 0.92)))

        # feedback message (ASCII only)
        if completed:
            msg = "COMPLETED - press ESC"
            surf_msg = font_hint.render(msg, True, (80, 220, 120))
            rect_msg = surf_msg.get_rect(center=(task_rect.centerx, int(h * 0.65)))
            screen.blit(surf_msg, rect_msg)
        else:
            if feedback == "wrong":
                msg = "Try again"
                surf_msg = font_hint.render(msg, True, (240, 90, 90))
                rect_msg = surf_msg.get_rect(center=(task_rect.centerx, int(h * 0.65)))
                screen.blit(surf_msg, rect_msg)
            elif feedback == "correct":
                msg = "Correct"
                surf_msg = font_hint.render(msg, True, (80, 220, 120))
                rect_msg = surf_msg.get_rect(center=(task_rect.centerx, int(h * 0.65)))
                screen.blit(surf_msg, rect_msg)

        # tree panel
        draw_tree_panel(
            screen,
            tree_rect,
            correct_by_level=correct_by_level,
            current_level=level,
            completed_levels=completed_levels,
            completed=completed
        )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
