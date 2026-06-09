# Steering Tuning Guide

This file documents the steering parameters used by the rule-based drive controller and where to change them.

## What Changed

The previous defaults reacted too late to lateral drift. With `K_STANLEY = 1.0`, `V_REF = 20.0`, and `STEER_DEADZONE = 0.03`, a moderate lane offset could produce a command close to the deadzone, so the wheel stayed nearly centered until the vehicle was already close to crossing the lane.

The current defaults make small lane offsets produce visible steering earlier and add a capped offset-rate term to reduce the hard correction feedback loop.

Main changes in `src/drive/mqtt_to_thrustmaster.py`:

| Parameter | Old | New | Why |
| --- | ---: | ---: | --- |
| `STEER_DEADZONE` | `0.03` | `0.01` | Keeps early small corrections instead of zeroing them. |
| `EMA_ALPHA` | `0.25` | `0.35` | Makes the target wheel angle react faster. |
| `K_STANLEY` | `1.0` | `2.0` | Stronger lateral-offset correction. |
| `V_REF` | `20.0` | `8.0` | Lower virtual speed makes Stanley offset correction more active. |
| `HISTORY_WINDOW_S` | `0.5` | `0.35` | Offset-rate damping reacts faster. |
| `K_D_OFFSET` | new | `0.20` | Adds early correction while drifting away and reduces correction while coming back. |
| `MAX_D_TERM_RAD` | new | `0.12` | Limits offset-rate spikes from noisy perception. |

## Where To Change Parameters

Primary runtime tuning is in:

```text
src/drive/mqtt_to_thrustmaster.py
```

Important sections:

| Location | Parameters |
| --- | --- |
| Lane assist toggle | `LANE_ASSIST_ENABLED_DEFAULT`, `LANE_ASSIST_TOGGLE_BUTTON` |
| Input filtering | `STEER_GAIN`, `STEER_DEADZONE`, `EMA_ALPHA`, `CMD_TIMEOUT_S`, `MIN_LANE_QUALITY` |
| Lateral controller | `MAX_STEER_RAD`, `K_STANLEY`, `V_REF`, `K_FF`, `HISTORY_WINDOW_S`, `K_D_OFFSET`, `MAX_D_TERM_RAD` |
| Wheel force feedback loop | `KP`, `KD`, `MAX_TORQUE`, `CONTROL_HZ` |

Controller implementation is in:

```text
src/drive/steering_controller.py
```

The optional Xbox bridge has its own final input smoothing in:

```text
src/drive/mqtt_to_xbox.py
```

## Lane Assist Toggle

Lane assist can be switched on and off while `mqtt_to_thrustmaster.py` is running.

Change the button index at the top of:

```text
src/drive/mqtt_to_thrustmaster.py
```

Relevant parameters:

| Parameter | Default | What it does |
| --- | ---: | --- |
| `LANE_ASSIST_ENABLED_DEFAULT` | `True` | Initial lane-assist state after startup. |
| `LANE_ASSIST_TOGGLE_BUTTON` | `0` | SDL button index used to toggle lane assist. Set to `-1` to disable button toggling. |

The button is edge-triggered: one press toggles once, and holding the button does not repeatedly toggle. When lane assist is off, the script keeps MQTT and diagnostics running but sends zero force-feedback steering torque so the driver can steer manually.

## Steering Formula

The lateral controller combines curvature feed-forward, Stanley-style feedback, and offset-rate damping:

```text
steer = K_FF * curvature_preview
      + heading_error_rad
      + atan2(K_STANLEY * offset_m, V_REF)
      + clamp(K_D_OFFSET * d_offset_dt, -MAX_D_TERM_RAD, MAX_D_TERM_RAD)
```

The result is limited to `MAX_STEER_RAD` and normalized to `steer_norm` in the range `-1.0..+1.0`.

## Parameter Reference

### Lateral Controller Parameters

| Parameter | Default | What it does | Increase when | Decrease when |
| --- | ---: | --- | --- | --- |
| `K_STANLEY` | `2.0` | Main gain for lateral lane offset. Higher means earlier/stronger correction away from lane edge. | Vehicle still drifts too close to lane boundary before correcting. | Vehicle reacts too strongly to small offset noise. |
| `V_REF` | `8.0` | Virtual speed used in the Stanley denominator. Lower value means stronger offset correction. | Steering still feels late. | Steering feels twitchy or too aggressive. |
| `K_D_OFFSET` | `0.20` | Uses offset change rate. If offset is growing, it adds correction earlier; if offset is shrinking, it backs off to reduce overshoot. | Vehicle still enters a delayed hard correction loop. | Steering reacts to noisy lane estimates or jitters. |
| `MAX_D_TERM_RAD` | `0.12` | Safety cap for the offset-rate contribution. | Offset-rate help is too weak but `K_D_OFFSET` is already useful. | Sudden steering jumps appear after noisy frames. |
| `HISTORY_WINDOW_S` | `0.35` | Time window used to estimate `d_offset_dt`. Shorter is more responsive, longer is smoother. | Rate damping feels too delayed. | Rate damping is noisy. |
| `K_FF` | `8.0` | Feed-forward from detected road curvature. Helps steer into curves before offset grows. | Curves are entered too late while straight-road behavior is fine. | Vehicle cuts into curves or oversteers on bends. |
| `MAX_STEER_RAD` | `0.5` | Maximum controller steering angle before normalization. | Command saturates too early and cannot recover. | Commands are too large for the simulator/wheel. |

### Input Filtering Parameters

| Parameter | Default | What it does | Increase when | Decrease when |
| --- | ---: | --- | --- | --- |
| `STEER_GAIN` | `1.0` | Multiplies final normalized steering before actuator filtering. | Whole system feels too weak after controller tuning. | Whole system feels too strong. |
| `STEER_DEADZONE` | `0.01` | Removes very small commands to avoid jitter. | Wheel jitters around center on straight roads. | Early corrections disappear or steering starts too late. |
| `EMA_ALPHA` | `0.35` | Target smoothing. Higher is more direct, lower is smoother. | Steering command feels delayed. | Steering oscillates because target changes too abruptly. |
| `CMD_TIMEOUT_S` | `0.25` | Centers wheel if lane messages stop. | Network/perception messages arrive with harmless small gaps. | Safety centering should happen sooner. |
| `MIN_LANE_QUALITY` | `0.3` | Minimum perception quality required to steer. | Bad lane detections cause steering jumps. | Controller stops steering too often despite usable lanes. |

### Wheel Force Feedback Parameters

| Parameter | Default | What it does | Increase when | Decrease when |
| --- | ---: | --- | --- | --- |
| `KP` | `0.35` | Wheel position proportional gain. Higher moves the physical wheel harder toward target. | Wheel physically lags far behind target. | Wheel overshoots or fights too hard. |
| `KD` | `0.08` | Wheel motor damping. Higher reduces physical overshoot. | Wheel overshoots target or oscillates mechanically. | Wheel feels sluggish or resists movement too much. |
| `MAX_TORQUE` | `0.5` | Force feedback torque limit. | Wheel cannot reach target quickly enough. | Wheel force is unsafe or uncomfortable. |
| `CONTROL_HZ` | `100` | Force feedback loop rate. | Usually do not change. | Usually do not change. |

## Suggested Test Procedure

Change one group at a time and test on the same track section.

1. If steering starts too late on straight roads, first lower `STEER_DEADZONE` or increase `K_STANLEY`.
2. If steering is still late, lower `V_REF` in small steps, for example `8.0 -> 6.0`.
3. If it corrects hard and then swings back, increase `K_D_OFFSET` slightly, for example `0.20 -> 0.25`, or lower `EMA_ALPHA` slightly, for example `0.35 -> 0.30`.
4. If perception noise causes twitching, increase `HISTORY_WINDOW_S`, lower `K_D_OFFSET`, or increase `STEER_DEADZONE` only as much as needed.
5. If only curves are late but straight-road centering is good, increase `K_FF` instead of `K_STANLEY`.

Useful diagnostics are published on `control/steering_cmd`:

| Field | Meaning |
| --- | --- |
| `steer_norm` | Raw normalized controller command before actuator deadzone/EMA. |
| `target_norm` | Final smoothed wheel target. |
| `wheel_norm` | Current physical wheel angle. |
| `wheel_error_norm` | Difference between target and physical wheel angle. |
| `offset_m` | Lateral lane offset used by the controller. |
| `d_offset_dt` | Estimated offset change rate used by damping. |
| `heading_err_rad` | Heading error from perception. |
| `curvature` | Curvature preview from perception. |
| `lane_assist_enabled` | Whether the lane assist is currently allowed to move the wheel. |
