import carla
import numpy as np
import cv2
import time
import random
import signal
import sys

# If True, we delete ALL vehicles.* at startup (best for your use-case to avoid stacking).
# If False, we only delete vehicles with role_name == "ego" or "npc".
NUKE_ALL_VEHICLES_ON_START = True

shutdown = False

def handle_exit(sig, frame):
    global shutdown
    if not shutdown:
        print("\nGraceful shutdown requested... (cleaning up)")
    shutdown = True

signal.signal(signal.SIGINT, handle_exit)   # Ctrl+C
signal.signal(signal.SIGTERM, handle_exit)


# == 1) connect ==
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
print("Connected to:", world.get_map().name)

bp_lib = world.get_blueprint_library()

def destroy_ids(actor_ids):
    if not actor_ids:
        return
    cmds = [carla.command.DestroyActor(x) for x in actor_ids]
    try:
        client.apply_batch_sync(cmds, True)
    except:
        pass

def cleanup_startup():
    # Controllers first (otherwise walkers can become “orphaned”)
    controllers = world.get_actors().filter("controller.ai.walker")
    ctrl_ids = [a.id for a in controllers]
    for a in controllers:
        try:
            a.stop()
        except:
            pass
    destroy_ids(ctrl_ids)

    walkers = world.get_actors().filter("walker.pedestrian.*")
    destroy_ids([a.id for a in walkers])

    vehicles = world.get_actors().filter("vehicle.*")
    if NUKE_ALL_VEHICLES_ON_START:
        destroy_ids([a.id for a in vehicles])
    else:
        tagged = []
        for a in vehicles:
            role = a.attributes.get("role_name")
            if role in ("ego", "npc"):
                tagged.append(a.id)
        destroy_ids(tagged)

    # Let UE / server process deletions
    try:
        world.wait_for_tick(seconds=0.2)
    except:
        pass

print("Startup cleanup...")
cleanup_startup()


# == 2a) spawn ego ==
vehicle_bp = bp_lib.find("vehicle.tesla.model3")
# Tag properly BEFORE spawning
try:
    vehicle_bp.set_attribute("role_name", "ego")
except:
    pass

spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
print("Spawned vehicle:", vehicle.type_id)

ego_loc = vehicle.get_transform().location
min_dist_from_ego = 25.0


# == 2b) spectator chase ==
spectator = world.get_spectator()

def follow_rear_chase(_):
    if shutdown:
        return
    t = vehicle.get_transform()
    spectator.set_transform(
        carla.Transform(
            t.location - t.get_forward_vector() * 8.0 + carla.Location(z=3.0),
            carla.Rotation(pitch=-15.0, yaw=t.rotation.yaw, roll=0.0)
        )
    )

world.on_tick(follow_rear_chase)


# == 3) traffic manager + autopilot ==
tm = client.get_trafficmanager(8000)
tm.set_synchronous_mode(False)
tm.set_global_distance_to_leading_vehicle(3.0)

vehicle.set_autopilot(True, tm.get_port())
print("Autopilot ON (Traffic Manager)")


# == 3b) spawn NPC vehicles ==
npc_vehicle_count = 35

vehicle_bps = bp_lib.filter("vehicle.*")
spawn_points = world.get_map().get_spawn_points()
random.shuffle(spawn_points)

npc_vehicles = []

for sp in spawn_points:
    if len(npc_vehicles) >= npc_vehicle_count:
        break

    if sp.location.distance(ego_loc) < min_dist_from_ego:
        continue

    bp = random.choice(vehicle_bps)

    if bp.has_attribute("number_of_wheels"):
        if int(bp.get_attribute("number_of_wheels").as_int()) < 4:
            continue

    if bp.has_attribute("color"):
        bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))

    # Tag properly BEFORE spawning
    try:
        bp.set_attribute("role_name", "npc")
    except:
        pass

    v = world.try_spawn_actor(bp, sp)
    if v is None:
        continue

    v.set_autopilot(True, tm.get_port())
    npc_vehicles.append(v)

print("Spawned NPC vehicles:", len(npc_vehicles))


# == 4) camera ==
camera_bp = bp_lib.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "1280")
camera_bp.set_attribute("image_size_y", "720")
camera_bp.set_attribute("fov", "90")

camera_transform = carla.Transform(carla.Location(x=1.6, z=1.4), carla.Rotation(pitch=-5))
camera = world.try_spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
if camera is None:
    raise RuntimeError("Failed to spawn camera sensor")
print("Camera attached")


# == 4b) pedestrians ==
def pick_nav_location(max_tries=25):
    for _ in range(max_tries):
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            return loc
    return None

pedestrian_count = 40

walker_bps = bp_lib.filter("walker.pedestrian.*")
walkers = []
walker_controllers = []

for _ in range(pedestrian_count):
    loc = pick_nav_location()
    if loc is None:
        continue

    wbp = random.choice(walker_bps)
    if wbp.has_attribute("is_invincible"):
        wbp.set_attribute("is_invincible", "false")

    w = world.try_spawn_actor(wbp, carla.Transform(loc))
    if w:
        walkers.append(w)

controller_bp = bp_lib.find("controller.ai.walker")

for w in walkers:
    c = world.spawn_actor(controller_bp, carla.Transform(), w)
    c.start()

    dest = pick_nav_location()
    if dest is not None:
        try:
            c.go_to_location(dest)
        except:
            pass

    if random.random() < 0.75:
        c.set_max_speed(random.uniform(1.4, 2.2))
    else:
        c.set_max_speed(random.uniform(2.5, 4.0))

    walker_controllers.append(c)

print("Spawned pedestrians:", len(walkers))

last_retarget_time = time.time()


# == camera callback ==
latest_image = None

def camera_callback(image):
    global latest_image
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    latest_image = arr[:, :, :3]

camera.listen(camera_callback)


# == writer ==
writer = cv2.VideoWriter(
    "camera_output.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    20,
    (1280, 720)
)
if not writer.isOpened():
    raise RuntimeError("VideoWriter failed to open")

print("Recording... click the 'Camera View' window then press 'q' (or Ctrl+C)")
cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)


def cleanup_end():
    # Stop controllers (best effort)
    for c in walker_controllers:
        try:
            c.stop()
        except:
            pass

    # Batch destroy everything we spawned (fast + reliable)
    destroy_ids([c.id for c in walker_controllers if c is not None])
    destroy_ids([w.id for w in walkers if w is not None])
    destroy_ids([v.id for v in npc_vehicles if v is not None])

    try:
        if camera is not None:
            try:
                camera.stop()
            except:
                pass
        destroy_ids([camera.id])
    except:
        pass

    try:
        destroy_ids([vehicle.id])
    except:
        pass

    # Extra sweep to prevent stacking even if something got missed
    cleanup_startup()


try:
    while not shutdown:
        # timeout so Ctrl+C always breaks out
        world.wait_for_tick(seconds=0.5)

        # keep pedestrians moving
        if time.time() - last_retarget_time > 6.0:
            last_retarget_time = time.time()
            for c in walker_controllers:
                dest = pick_nav_location()
                if dest is not None:
                    try:
                        c.go_to_location(dest)
                    except:
                        pass

        if latest_image is not None:
            cv2.imshow("Camera View", latest_image)
            writer.write(latest_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Q pressed, stopping...")
            break

except Exception as e:
    print("Runtime error:", e)

finally:
    print("Cleanup starting...")
    cleanup_end()

    try:
        writer.release()
    except:
        pass

    try:
        cv2.destroyAllWindows()
    except:
        pass

    print("Cleanup done. Saved camera_output.avi")
    sys.exit(0)