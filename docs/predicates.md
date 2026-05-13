# Predicate Reference

This document defines the predicates used in the neuro-symbolic pipeline.

The predicates are divided into:
- input predicates (produced by the perception module)
- output predicates (defined by the user through ASP rules)

## Time semantics

In this implementation, DP-SR timepoints correspond to sequential video frame
numbers, assigned by the agent starting from `0`. At each reasoning step, `@now`
denotes the current frame number.

When an output predicate contains temporal arguments, `T` denotes the frame
number in which the anomaly or candidate anomaly occurs, while `T1` and `T2`
denote the inclusive bounds of a frame interval.

## Input predicates

These predicates are generated automatically at each reasoning step.

### `object(ID, Class, X, Y)`
Represents a detected object in the given frame.

- `ID`: tracking identifier assigned by the detector
- `Class`: predicted object class
- `X`, `Y`: coordinates of the center of the bounding box

The coordinate system follows the standard image convention:
- origin at the top-left corner
- X increases from left to right
- Y increases from top to bottom

The identifier `ID` is intended to remain stable across consecutive frames, allowing object tracking over time.

---

### `area(AreaID, AreaType)`
Represents a semantic region defined by the user.

- `AreaID`: unique identifier of the region
- `AreaType`: semantic label

Examples of area types include:
- `sidewalk`
- `bikelane`
- `doorway`

This predicate is provided only if the user defines regions.

---

### `in_area(AreaID, ObjectID)`

Represents the spatial relation between an object and a region.

- `AreaID`: region identifier
- `ObjectID`: object identifier

Semantics:
the object is considered inside the region if the **centroid of its bounding box** falls within the region.

> [!WARNING]
> This relation is based on a geometrical approximation. Since it relies only on the bounding box centroid, it may produce misclassifications in some cases, especially when the object is only partially inside a region, when regions overlap, or when perspective distortion affects the image.

---

## Output predicates

These predicates are not provided as input.  
They must be **derived through user-defined ASP rules**.

---

### `anomaly(ID, AnomalyType, T)`

Represents a **single-frame confirmed anomaly**.

- `ID`: identifier of the involved entity
- `AnomalyType`: label of the anomaly
- `T`: frame number at which the anomaly occurs. Use `@now` to refer to the current frame.

Use this predicate when the anomaly can be fully determined using symbolic reasoning alone.

---

### `anomaly(ID, AnomalyType, T1, T2)`

Represents a **temporal confirmed anomaly** over an interval.

- `ID`: identifier of the involved entity
- `AnomalyType`: label of the anomaly
- `T1`: first frame number of the interval
- `T2`: last frame number of the interval, with `T1 <= T2`

Use this predicate when the confirmed anomaly depends on temporal behavior and does not require VLM-based validation.

---

### `candidate_anomaly(ID, AnomalyType, Description, T)`

Represents a **single-frame candidate anomaly**.

- `ID`: identifier of the involved entity
- `AnomalyType`: label of the candidate anomaly
- `Description`: natural language description of the candidate anomaly
- `T`: frame number. Use `@now` to refer to the current frame.

Use this predicate when:
- the situation may be anomalous
- symbolic reasoning is not sufficient to confirm it

The VLM receives the frame identified by `T` together with `Description`.

---

### `candidate_anomaly(ID, AnomalyType, Description, T1, T2)`

Represents a **temporal candidate anomaly** over an interval.

- `ID`: identifier of the involved entity
- `AnomalyType`: label of the candidate anomaly
- `Description`: natural language description of the candidate anomaly
- `T1`: first frame number of the interval
- `T2`: last frame number of the interval, with `T1 <= T2`

Use this predicate when the anomaly depends on temporal behavior.

The VLM receives visual evidence extracted from the inclusive interval `[T1, T2]` together with `Description`.

Examples:
- loitering
- suspicious inactivity

---

## Meaning of `ID`

The first argument of `anomaly/3`, `anomaly/4`, `candidate_anomaly/4`, and
`candidate_anomaly/5` identifies the entity involved in the anomaly.

In most examples, anomalies are expressed with respect to detected objects.
Therefore, this argument is often an object identifier coming from `object/4`.

However, its semantics is intentionally left to the user-defined rules. It can
refer to any entity that is meaningful for the program, such as an object, an area, or a frame. For instance,
a frame-level anomaly may use an identifier such as `frame` to indicate that the
anomaly concerns the frame as a whole rather than a specific tracked object.
