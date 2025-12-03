# mario_motion_clean_embedded_cam.py
# Single-file: Mario (boss) + Webcam control (Mediapipe + OpenCV fallback) + Embedded camera UI in Pygame
# - Embedded camera preview (CAM_W x CAM_H) shown in top-right corner of the game window
# - Mario cannot move left past the starting x (120)
# - Punch = FIRE + JUMP
#
# Requirements: pygame, opencv-python, mediapipe
# pip install pygame opencv-python mediapipe

import pygame, sys, time, threading, math, random
import cv2
import numpy as np

# Try to import mediapipe
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# ---------------- PYGAME SETUP -------------
pygame.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mario — Motion Control (Embedded Cam)")
clock = pygame.time.Clock()
FONT_L = pygame.font.SysFont("Tahoma", 36)
FONT_M = pygame.font.SysFont("Tahoma", 24)
FONT_S = pygame.font.SysFont("Tahoma", 18)

# Colors
WHITE=(255,255,255); BLACK=(0,0,0)
SKY=(120,190,255); PLATFORM=(100,60,30); ORANGE=(255,140,0)
DARK=(18,18,18)

# ---------------- PIXEL SPRITES (procedural) -------------
def sprite(pattern, scale=4, cmap=None):
    h=len(pattern); w=len(pattern[0])
    surf = pygame.Surface((w*scale, h*scale), pygame.SRCALPHA)
    default = {'r':(220,40,40),'o':(255,150,40),'y':(255,210,60),'g':(60,180,80),
               'k':(20,20,20),'w':(240,240,240),'b':(40,120,220),'p':(255,120,200)}
    cmap = cmap or default
    for y,row in enumerate(pattern):
        for x,ch in enumerate(row):
            if ch!='.':
                c = cmap.get(ch,(255,255,255))
                pygame.draw.rect(surf, c, (x*scale,y*scale,scale,scale))
    return surf

PLAYER_SMALL = sprite(["..rr..",".rrrr.",".rrrr.", "..rr.."], scale=8)
PLAYER_BIG   = sprite(["..rr..",".rrrr.",".rrrr.",".rrrr.",".rrrr.", "..rr.."], scale=8)
PLAYER_FIRE  = sprite(["..oo..",".oooo.",".oooo.", "..oo.."], scale=8)
GOOMBA       = sprite([".kkkk.","kkkkkk",".kkkk.", ".k..k."], scale=6)
COIN_SPR     = sprite([".yyy.","yyyyy",".yyy."], scale=6)
MUSH_SPR     = sprite([".rrr.","rrrrr",".rrr.","..r.."], scale=7)
FIREBALL_SPR = sprite([".ooo.","ooooo",".ooo."], scale=5)
BOSS_SPR     = sprite([".ppppp.","pprbbpp","pprbbpp",".ppppp."], scale=12)

# ---------------- WORLD (single large level) -------------
LEVEL_LENGTH = 7000
TILE = 40
platforms = []
def add_platform(x,y,w): platforms.append(pygame.Rect(x,y,w,24))

add_platform(0, HEIGHT-120, LEVEL_LENGTH)

for i in range(10,160,10):
    x = i*40
    if 6 <= (i % 12) <= 8:
        add_platform(x, HEIGHT-220 - ((i%5)*8), 160)
    elif (i%7)==0:
        add_platform(x, HEIGHT-320 - ((i%4)*12), 120)
    elif (i%5)==0:
        add_platform(x, HEIGHT-420 - ((i%6)*10), 100)

add_platform(1800, HEIGHT-300, 220)
add_platform(2300, HEIGHT-380, 180)
add_platform(3000, HEIGHT-260, 260)
add_platform(3900, HEIGHT-340, 220)
add_platform(4700, HEIGHT-300, 240)
add_platform(5600, HEIGHT-360, 200)

# coins, enemies, powerups, blocks, lava, flag, boss zone
coins = []
for x in range(600, LEVEL_LENGTH-800, 300):
    for i in range(3):
        coins.append(pygame.Rect(x + i*36, HEIGHT-220 - (i%2)*40, 24, 24))

goombas = [{'rect': pygame.Rect(gx, HEIGHT-160, 34, 34), 'dir': -1, 'spd': 1.2}
           for gx in [700, 1400, 1900, 2600, 3400, 4200, 5000, 5800]]

powerups = [
    ("mushroom", pygame.Rect(1200, HEIGHT-260, 28, 28)),
    ("mushroom", pygame.Rect(3600, HEIGHT-420, 28, 28)),
    ("flower",   pygame.Rect(5200, HEIGHT-340, 28, 28))
]

blocks = [
    pygame.Rect(1500, HEIGHT-300, 40, 40),
    pygame.Rect(2800, HEIGHT-340, 40, 40),
    pygame.Rect(4400, HEIGHT-380, 40, 40)
]

lava = [pygame.Rect(6100, HEIGHT-80, 300, 80)]
flag_rect = pygame.Rect(LEVEL_LENGTH-200, HEIGHT-320, 48, 200)
boss_zone_x = LEVEL_LENGTH-1200

# ---------------- PLAYER & GAME STATE -------------
player = pygame.Rect(120, HEIGHT-200, 34, 50)
vel_y = 0; on_ground=False; speed=6; jump_power=-15; last_dir=1
form = "fire"
shoot_cooldown = 0.35; last_shot = -10
score = 0; lives = 3
fireballs = []
enemy_projectiles = []

boss = {
    'rect': pygame.Rect(LEVEL_LENGTH-350, HEIGHT-420, 220, 260),
    'hp': 20,
    'max_hp': 20,
    'dir': -1,
    'spd': 1.2,
    'last_shot': 0,
    'alive': True
}

cam_x = 0
GAME_MENU="menu"; GAME_PLAY="play"; GAME_PAUSE="pause"; GAME_WIN="win"; GAME_OVER="over"
game_state = GAME_MENU

# ---------------- CONTROL STATE SHARED -------------
cs_lock = threading.Lock()
control_state = {
    "enabled": True,
    "move": 0,
    "shoot": False,
    "jump": False,
    "last_update": time.time(),
    "cam_frame": None
}

# ---------------- WEBCAM PARAMETERS -------------
CAM_W, CAM_H = 500, 400
CAM_MARGIN = 12
CAM_POS = (WIDTH - CAM_W - CAM_MARGIN, CAM_MARGIN)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

bg_avg = None
last_centroid_x = None
last_centroid_time = 0
last_centroid = None

if MP_AVAILABLE:
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
else:
    hands_detector = None

MOTION_AREA_THRESHOLD = 1000
PUNCH_DX_THRESHOLD = 20
OPEN_PALM_FINGERS = 4
HAND_RAISE_Y = 0.45


# --- Webcam Thread ---------------------------------------------------
def webcam_loop():
    global bg_avg, last_centroid_x, last_centroid_time, last_centroid
    last_punch_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # --- Motion Fallback ------------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21,21), 0)

        motion_move = 0
        motion_area = 0
        motion_dx = 0

        if bg_avg is None:
            bg_avg = gray.astype("float")
        else:
            cv2.accumulateWeighted(gray, bg_avg, 0.12)
            diff = cv2.absdiff(gray, cv2.convertScaleAbs(bg_avg))
            _, thr = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            thr = cv2.dilate(thr, None, 2)
            cnt, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if cnt:
                largest = max(cnt, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                motion_area = area

                if area > MOTION_AREA_THRESHOLD:
                    M = cv2.moments(largest)
                    if M["m00"] != 0:
                        cx = int(M["m10"]/M["m00"])

                        if last_centroid_x is not None:
                            dx = cx - last_centroid_x
                            motion_dx = dx
                            if dx > 8: motion_move = 1
                            elif dx < -8: motion_move = -1

                        last_centroid_x = cx
                        last_centroid_time = time.time()

        # --- Mediapipe ----------------------------------------------
        mp_move = 0
        mp_shoot = False
        mp_jump = False
        hands_count = 0

        if MP_AVAILABLE and hands_detector:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands_detector.process(rgb)

            if result.multi_hand_landmarks:
                hands_count = len(result.multi_hand_landmarks)
                infos = []

                for hand in result.multi_hand_landmarks:
                    pts = [(int(lm.x*w), int(lm.y*h)) for lm in hand.landmark]
                    wrist_y = pts[0][1]

                    def ext(tip, pip): return pts[tip][1] < pts[pip][1]
                    fingers = 0
                    if ext(8,6): fingers+=1
                    if ext(12,10): fingers+=1
                    if ext(16,14): fingers+=1
                    if ext(20,18): fingers+=1
                    if pts[4][0] > pts[3][0]: fingers+=1

                    centroid_x = sum(p[0] for p in pts)/len(pts)
                    infos.append({"x":centroid_x, "y":wrist_y, "f":fingers})

                avg_x = sum(i["x"] for i in infos)/len(infos)

                if avg_x > w*0.55: mp_move = 1
                elif avg_x < w*0.45: mp_move = -1

                if any(i["f"] >= OPEN_PALM_FINGERS for i in infos):
                    mp_shoot = True

                if len(infos)>=2 and all(i["y"] < h*HAND_RAISE_Y for i in infos):
                    mp_jump = True

                # --- Punch = FIRE + JUMP -----------------------------------
                if last_centroid is not None:
                    dx_avg = avg_x - last_centroid
                    if abs(dx_avg)>PUNCH_DX_THRESHOLD and (time.time()-last_centroid_time)<0.25:
                        if time.time()-last_punch_time>0.25:
                            mp_shoot = True
                            mp_jump  = True     # JUMP ADDED!!!
                            last_punch_time = time.time()

                last_centroid = avg_x
                last_centroid_time = time.time()

        # --- Fusing hand + motion ------------------------------------
        with cs_lock:
            if not control_state["enabled"]:
                control_state["move"]=0
                control_state["shoot"]=False
                control_state["jump"]=False
            else:
                if hands_count>0:
                    control_state["move"] = mp_move
                    control_state["shoot"]= mp_shoot
                    control_state["jump"] = mp_jump
                else:
                    control_state["move"] = motion_move
                    control_state["shoot"]= (motion_area>7000) or (abs(motion_dx)>30)
                    control_state["jump"] = False

                control_state["last_update"] = time.time()

        # --- UI frame for embedding in pygame -------------------------
        ui = cv2.resize(frame, (CAM_W, CAM_H))
        overlay = ui.copy()

        with cs_lock:
            mv = control_state["move"]
            sh = control_state["shoot"]
            ju = control_state["jump"]
            en = control_state["enabled"]

        colL = (0,255,0) if mv==-1 else (200,200,200)
        colR = (0,255,0) if mv==1 else (200,200,200)

        cv2.arrowedLine(overlay,(50,CAM_H//2),(120,CAM_H//2),colL,8)
        cv2.arrowedLine(overlay,(CAM_W-50,CAM_H//2),(CAM_W-120,CAM_H//2),colR,8)

        if sh:
            cv2.putText(overlay,"FIRE",(CAM_W//2-40,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,255),3)
        if ju:
            cv2.putText(overlay,"JUMP",(CAM_W//2-40,CAM_H-20),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,200,255),3)

        status = "CAM ON" if en else "CAM OFF"
        cv2.putText(overlay,status,(8,20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        ui = cv2.addWeighted(overlay,0.9,ui,0.1,0)

        with cs_lock:
            control_state["cam_frame"] = ui.copy()

# thread start
threading.Thread(target=webcam_loop,daemon=True).start()

# ---------------- Helper Functions -----------------------------------
def draw_text_center(txt, y, font=FONT_M, color=WHITE):
    surf = font.render(txt, True, color)
    screen.blit(surf, (WIDTH//2 - surf.get_width()//2, y))

def cam_rect(r):
    return pygame.Rect(r.x-cam_x, r.y, r.w, r.h)

def reset_game():
    global player, vel_y, form, score, lives
    global fireballs, enemy_projectiles, boss, cam_x, goombas

    player.x, player.y = 120, HEIGHT-200
    vel_y = 0
    form = "fire"
    score = 0
    lives = 3
    fireballs.clear()
    enemy_projectiles.clear()

    boss["hp"] = boss["max_hp"]
    boss["alive"] = True
    boss["rect"].x = LEVEL_LENGTH-350

    cam_x = 0

    gx_list = [700,1400,1900,2600,3400,4200,5000,5800]
    goombas.clear()
    for gx in gx_list:
        goombas.append({
            "rect": pygame.Rect(gx,HEIGHT-160,34,34),
            "dir": -1,
            "spd": 1.2
        })

reset_game()

# ---------------- MAIN LOOP ------------------------------------------
running = True
last_shot_time = 0

while running:
    dt = clock.tick(60)/1000.0
    now = time.time()

    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False

        if ev.type == pygame.KEYDOWN:
            if game_state == GAME_MENU and ev.key == pygame.K_RETURN:
                game_state = GAME_PLAY

            elif game_state == GAME_PLAY:
                if ev.key == pygame.K_p:
                    game_state = GAME_PAUSE

                if ev.key == pygame.K_c or ev.key == pygame.K_q:
                    with cs_lock:
                        control_state["enabled"] = not control_state["enabled"]

                if ev.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                    if form=="fire" and now-last_shot_time > shoot_cooldown:
                        last_shot_time = now
                        fb = {
                            "rect": pygame.Rect(player.centerx+(last_dir*20),
                                                player.centery-8, 18,18),
                            "vx": 16*last_dir
                        }
                        fireballs.append(fb)

                if ev.key==pygame.K_SPACE and on_ground:
                    vel_y = jump_power

            elif game_state == GAME_PAUSE:
                if ev.key == pygame.K_p:
                    game_state = GAME_PLAY

            elif game_state in (GAME_WIN, GAME_OVER):
                if ev.key == pygame.K_RETURN:
                    reset_game()
                    game_state = GAME_MENU

    keys = pygame.key.get_pressed()

    # ---------------- MENU ----------------------------
    if game_state == GAME_MENU:
        screen.fill(DARK)
        draw_text_center("MEGA MARIO — MOTION CONTROL", 140, FONT_L)
        draw_text_center("Press ENTER to Start", 260, FONT_M)
        draw_text_center("Toggle Camera Control: C or Q", 320, FONT_S)
        draw_text_center("A/D Move | Space Jump | Shift Fire", 360, FONT_S)
        pygame.display.flip()
        continue

    # ---------------- PAUSE ---------------------------
    if game_state == GAME_PAUSE:
        draw_text_center("PAUSED — Press P to Resume", HEIGHT//2, FONT_L)
        pygame.display.flip()
        continue

    # ---------------- PLAY ----------------------------
    if game_state == GAME_PLAY:

        # Get latest gesture controls
        with cs_lock:
            cs = control_state.copy()

        move_cmd = 0
        shoot_cmd = False
        jump_cmd = False

        if cs["enabled"] and (now - cs["last_update"] < 1.0):
            move_cmd = cs["move"]
            shoot_cmd = cs["shoot"]
            jump_cmd  = cs["jump"]

        # keyboard overrides
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:  move_cmd = -1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: move_cmd = 1
        if keys[pygame.K_SPACE]: jump_cmd = True
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shoot_cmd = True

        # ---- Movement ----
        dx = 0
        if move_cmd==-1: dx=-speed; last_dir=-1
        if move_cmd==1:  dx= speed; last_dir= 1

        player.x += dx

        # prevent going backwards
        if player.x < 120:
            player.x = 120

        for p in platforms:
            if player.colliderect(p):
                if dx>0: player.right = p.left
                elif dx<0: player.left  = p.right

        # ---- Jump ----
        if jump_cmd and on_ground:
            vel_y = jump_power

        # ---- Gravity ----
        vel_y += 40*dt
        player.y += vel_y
        on_ground = False

        for p in platforms:
            if player.colliderect(p):
                if vel_y>0 and player.bottom <= p.top+30:
                    player.bottom = p.top
                    vel_y = 0
                    on_ground = True

                elif vel_y<0 and player.top >= p.bottom-30:
                    player.top = p.bottom
                    vel_y = 0

                    # break block
                    for b in blocks[:]:
                        if player.top <= b.y+8 and b.x <= player.centerx <= b.x+b.w:
                            import random
                            if random.random()<0.5:
                                coins.append(pygame.Rect(b.x,b.y-48,22,22))
                            else:
                                powerups.append(("mushroom",
                                                pygame.Rect(b.x,b.y-40,26,26)))
                            try: blocks.remove(b)
                            except: pass

        # ---- Collect coins ----
        for c in coins[:]:
            if player.colliderect(c):
                coins.remove(c)
                score+=100

        # ---- Powerups ----
        for pu in powerups[:]:
            typ, r = pu
            if player.colliderect(r):
                powerups.remove(pu)
                if typ=="mushroom": form="big"
                elif typ=="flower": form="fire"
                score += 500

        # ---- Shoot ----
        if shoot_cmd and form=="fire" and now-last_shot_time > shoot_cooldown:
            last_shot_time = now
            fireballs.append({
                "rect": pygame.Rect(player.centerx+(last_dir*20),
                                    player.centery-8,18,18),
                "vx": 16*last_dir
            })

        # ---- Update Fireballs ----
        for fb in fireballs[:]:
            fb["rect"].x += fb["vx"]

            for g in goombas[:]:
                if fb["rect"].colliderect(g["rect"]):
                    goombas.remove(g)
                    fireballs.remove(fb)
                    score+=200
                    break

            if boss["alive"] and fb["rect"].colliderect(boss["rect"]):
                boss["hp"]-=1
                fireballs.remove(fb)
                score+=50
                if boss["hp"]<=0:
                    boss["alive"]=False
                    score+=2000

            if fb["rect"].x < cam_x-200 or fb["rect"].x > cam_x+WIDTH+200:
                fireballs.remove(fb)

        # ---- Goombas ----
        for g in goombas[:]:
            g["rect"].x += g["dir"] * g["spd"]

            foot = pygame.Rect(
                g["rect"].x + g["dir"]*10,
                g["rect"].bottom+4,
                g["rect"].w,
                4
            )
            if not any(foot.colliderect(p) for p in platforms):
                g["dir"] *= -1

            if player.colliderect(g["rect"]):
                if vel_y>1:
                    goombas.remove(g)
                    vel_y = jump_power/2
                    score+=200
                else:
                    lives-=1
                    player.x = max(120,player.x-200)
                    player.y = HEIGHT-200
                    vel_y = 0
                    if lives<=0:
                        game_state = GAME_OVER

        # ---- Boss ----
        if player.x > boss_zone_x and boss["alive"]:
            bx = boss["rect"].x

            if bx > LEVEL_LENGTH-200: boss["dir"] = -1
            if bx < LEVEL_LENGTH-700: boss["dir"] = 1

            boss["rect"].x += boss["dir"]*boss["spd"]

            if now-boss["last_shot"] > 1.6:
                boss["last_shot"] = now
                d = 1 if player.centerx > boss["rect"].centerx else -1
                enemy_projectiles.append({
                    "rect": pygame.Rect(
                        boss["rect"].centerx + d*30,
                        boss["rect"].centery,
                        18,18
                    ),
                    "vx": d*8,
                    "vy": -6
                })

        # ---- Boss projectiles ----
        for proj in enemy_projectiles[:]:
            proj["rect"].x += proj["vx"]
            proj["rect"].y += proj["vy"]
            proj["vy"] += 0.4

            if proj["rect"].colliderect(player):
                enemy_projectiles.remove(proj)
                lives-=1
                if lives<=0:
                    game_state = GAME_OVER

            if proj["rect"].y>HEIGHT+200:
                enemy_projectiles.remove(proj)

        # ---- Boss stomp ----
        if boss["alive"] and player.colliderect(boss["rect"]):
            if vel_y>1 and player.bottom - boss["rect"].top < 40:
                boss["hp"]-=2
                vel_y = jump_power/1.8
                if boss["hp"]<=0:
                    boss["alive"] = False
                    score+=2000
            else:
                lives-=1
                player.x = max(120,player.x-300)
                player.y = HEIGHT-200
                vel_y = 0
                if lives<=0:
                    game_state = GAME_OVER

        # ---- Win Condition ----
        if not boss["alive"] and player.x > flag_rect.x - 100:
            game_state = GAME_WIN

        # ---- Camera ----
        cam_x += (player.x - cam_x - WIDTH//2)*min(1,8*dt)
        cam_x = max(0,min(cam_x,LEVEL_LENGTH-WIDTH))

        # ---------------- DRAW EVERYTHING ----------------------------
        screen.fill(SKY)

        # sky
        for i in range(8):
            cx = (i*300 - cam_x*0.15)%(WIDTH+200)-100
            pygame.draw.ellipse(screen,WHITE,(cx,60+(i%2)*20,180,50))

        # platforms
        for p in platforms:
            pygame.draw.rect(screen,PLATFORM,cam_rect(p))
            pygame.draw.rect(screen,(80,50,30),cam_rect(p),2)

        # lava
        for L in lava:
            pygame.draw.rect(screen,ORANGE,cam_rect(L))

        # blocks
        for b in blocks:
            pygame.draw.rect(screen,(200,160,70),cam_rect(b))
            pygame.draw.rect(screen,(120,80,40),cam_rect(b),2)

        # coins
        for c in coins:
            screen.blit(COIN_SPR,cam_rect(c))

        # powerups
        for typ,r in powerups:
            if typ=="mushroom":
                screen.blit(MUSH_SPR,cam_rect(r))
            else:
                screen.blit(PLAYER_FIRE,cam_rect(r))

        # goombas
        for g in goombas:
            screen.blit(GOOMBA,cam_rect(g["rect"]))

        # fireballs
        for fb in fireballs:
            screen.blit(FIREBALL_SPR,cam_rect(fb["rect"]))

        # boss projectiles
        for proj in enemy_projectiles:
            screen.blit(FIREBALL_SPR,cam_rect(proj["rect"]))

        # boss
        if boss["alive"]:
            screen.blit(BOSS_SPR,cam_rect(boss["rect"]))
            bx = 160; by = 16; bw = WIDTH-320; bh=20
            pygame.draw.rect(screen,(80,80,80),(bx,by,bw,bh))
            ratio = boss["hp"]/boss["max_hp"]
            pygame.draw.rect(screen,(180,30,120),(bx,by,int(bw*ratio),bh))
            screen.blit(FONT_S.render("BOSS",True,WHITE),(bx-60,12))

        # flag
        fr = cam_rect(flag_rect)
        pygame.draw.rect(screen,(180,100,40),(fr.x+10,fr.y,6,fr.h))
        pygame.draw.rect(screen,(240,60,60),(fr.x+16,fr.y+10,60,40))
        pygame.draw.rect(screen,(0,0,0),(fr.x+16,fr.y+10,60,40),2)

        # player
        ps = PLAYER_SMALL if form=="small" else PLAYER_BIG if form=="big" else PLAYER_FIRE
        screen.blit(ps,cam_rect(player))

        # HUD
        screen.blit(FONT_S.render(f"Score: {score}",True,BLACK),(16,16))
        screen.blit(FONT_S.render(f"Lives: {lives}",True,BLACK),(16,40))

        with cs_lock:
            en = control_state["enabled"]
        screen.blit(FONT_S.render("CAM ON" if en else "CAM OFF",True,BLACK),(16,64))

        # ---- CAMERA EMBED ----
        with cs_lock:
            cam_frame = control_state["cam_frame"]

        if cam_frame is not None:
            try:
                rgb = cv2.cvtColor(cam_frame,cv2.COLOR_BGR2RGB)
                cam_surf = pygame.image.frombuffer(rgb.tobytes(),
                                                   (CAM_W,CAM_H),"RGB")
                bg = pygame.Surface((CAM_W+4,CAM_H+4))
                bg.fill((40,40,40))
                screen.blit(bg,(CAM_POS[0]-2,CAM_POS[1]-2))
                screen.blit(cam_surf,CAM_POS)
            except:
                pass

        pygame.display.flip()
        continue

    # ---------------- WIN ---------------------------------------
    if game_state == GAME_WIN:
        screen.fill((20,120,80))
        draw_text_center("YOU DEFEATED THE BOSS!",200,FONT_L)
        draw_text_center(f"Score: {score}",260,FONT_M)
        draw_text_center("Press ENTER to return to menu",330,FONT_S)
        pygame.display.flip()
        continue

    # ---------------- GAME OVER ---------------------------------
    if game_state == GAME_OVER:
        screen.fill((80,10,10))
        draw_text_center("GAME OVER",200,FONT_L)
        draw_text_center("Press ENTER to retry",260,FONT_M)
        pygame.display.flip()
        continue

# ------------ EXIT CLEANUP -------------------------------------
pygame.quit()
cap.release()
if MP_AVAILABLE and hands_detector:
    hands_detector.close()
cv2.destroyAllWindows()
sys.exit()
