"""Demonstrates planning and simulation pickling with and without GP cache reading."""
from frontmatter_crossflow import *

# Experiment name
PLAN_NO_CACHE = True
PLAN_FROM_CACHE = False

if PLAN_NO_CACHE:
    experiment_name = "crossflow_planner_nocache"

    ##################################################
    # Create model
    ##################################################
    # Create Model
    mtt = Crossflow(plume_loc=(0, 0, 0), extent=extent, s=s,
                tprof=tprof, sprof=sprof,
                rhoprof=rhoprof, vex=v0_param, area=a0_param,
                density=rho0, curfunc=curmag, headfunc=curhead,
                salt=s0, temp=t0, E=(alph_param, bet_param))

    ##################################################
    # Create trajectory optimization object
    ##################################################
    print("Generating trajectory optimizer...")
    # Trajectory planning
    planners = []
    budget = time_resolution * vel  # distance budget per leg
    for t, start_time in enumerate(np.arange(0, duration, step=time_resolution)):
        print(f"Initializing plan {t}")
        # Create the base trajectory generator object with a temporary start_point
        # and t0 value. We will reset these later in the chaining process.
        traj_generator = LawnSpiralWithStartGeneratorFlexible(
            t0=start_time, vel=vel, alt=alt, start_point=(0., 0.),
            traj_type=traj_type, res=traj_res)

        # Create planner
        planners.append(TrajectoryOpt(
            mtt,
            traj_generator,
            reward,
            x0=(1000., 700., 135., 0., 0.),  # (lh, lw, rot, origin_x, origin_y)
            #param_bounds=[(20., 100), (20., 100.), (-360., 360.), extent.xrange, extent.yrange],
            param_names={"lh": 0, "lw": 1, "rot": 2, "origin_x": 3, "origin_y": 4},
            budget=budget,
            limits=[extent.xmin, extent.xmax, extent.ymin, extent.ymax],
            max_iters=0,
            tol=1e-2
        ))

    planner = TrajectoryChain(planners=planners)

    ##################################################
    # Get trajectory to execute
    ##################################
    print("Getting trajectory...")
    tic()
    plan_opt = planner.get_plan()
    toc()

    ##################################################
    # Execute Trajectory in Real Environment to get Obs
    ##################################################
    print("Executing trajectory in real environment...")

    # Create the robot
    rob = OfflineRobot(mtt, plan_opt, env, vel, com_window)

    # Create the simulator
    simulator = Simulator(rob, env, ref_global=True)

    # Run the simulator
    times = np.arange(0, duration + 1, 100)
    tic()
    simulator.simulate(times, experiment_name=experiment_name)
    toc()

    # Observations
    n_pts = 20  # number of time points to simulate
    times = np.linspace(0, 3600 * 12, 10)
    fig_world, fig_global = simulator.plot_world3d(times=times)
    fig_world.show()
    fig_global.show()

    # Save mission state
    mission_hash = get_mission_hash(experiment_name)
    save_mission(mission_hash, rob, mtt, env, simulator)
    print("Saved full simulation to", mission_hash)


if PLAN_GP_CACHE:
    experiment_name = "crossflow_planner_gpcache"

    ##################################################
    # Create model
    ##################################################
    mtt2 = Crossflow(plume_loc=(0, 0, 0), extent=extent, s=s,
                tprof=tprof, sprof=sprof,
                rhoprof=rhoprof, vex=v0_param, area=a0_param,
                density=rho0, curfunc=curmag, headfunc=curhead,
                salt=s0, temp=t0, E=(alph_param, bet_param))

    ##################################################
    # Create trajectory optimization object
    ##################################################
    print("Generating trajectory optimizer...")
    # Trajectory planning
    planners = []
    budget = time_resolution * vel  # distance budget per leg
    for t, start_time in enumerate(np.arange(0, duration, step=time_resolution)):
        print(f"Initializing plan {t}")
        # Create the base trajectory generator object with a temporary start_point
        # and t0 value. We will reset these later in the chaining process.
        traj_generator = LawnSpiralWithStartGeneratorFlexible(
            t0=start_time, vel=vel, alt=alt, start_point=(0., 0.),
            traj_type=traj_type, res=traj_res)

        # Create planner
        planners.append(TrajectoryOpt(
            mtt2,
            traj_generator,
            reward,
            x0=(75., 75., 0., 0., 0.),  # (lh, lw, rot, origin_x, origin_y)
            param_bounds=[(20., 100), (20., 100.), (-360., 360.), extent.xrange, extent.yrange],
            param_names={"lh": 0, "lw": 1, "rot": 2, "origin_x": 3, "origin_y": 4},
            budget=budget,
            limits=[extent.xmin, extent.xmax, extent.ymin, extent.ymax],
            max_iters=1,
            tol=1e-2
        ))

    planner = TrajectoryChain(planners=planners)

    ###################
    # Writing the GP cache
    ###################
    tic()
    print("Writing GP cache.")
    mtt.write_gp_cache(tvec=[p.traj_generator.t0 for p in planners],
                    xrange=[-500, 500], xres=20,
                    yrange=[-500, 500], yres=20,
                    zrange=[0, 100], zres=5,
                    overwrite=True,
                    visualize=True)
    print("Done.")
    toc()

    ##################################################
    # Get trajectory to execute
    ##################################
    print("Getting trajectory...")
    tic()
    plan_opt = planner.get_plan(from_cache=True)
    toc()

    ##################################################
    # Execute Trajectory in Real Environment to get Obs
    ##################################################
    print("Executing trajectory in real environment...")

    # Create the robot
    rob = OfflineRobot(mtt, plan_opt, env, vel, com_window)

    # Create the simulator
    simulator = Simulator(rob, env, ref_global=True)

    # Run the simulator
    times = np.arange(0, duration + 1, 10)
    tic()
    simulator.simulate(times, experiment_name=experiment_name)
    toc()

    # Save mission state
    mission_hash = get_mission_hash(experiment_name)
    save_mission(mission_hash, rob, mtt, env, simulator)
    print("Saved full simulation to", mission_hash)
