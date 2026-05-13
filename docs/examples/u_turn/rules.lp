
% =========================
% Static knowledge
% =========================

vehicle(car). vehicle(truck). vehicle(motorcycle). vehicle(bus).

% An object is a vehicle if its detected class belongs
% to the set of vehicle classes.
#temp is_vehicle(ID) :- object(ID, Class, _, _), vehicle(Class).

vehicle_in_traffic_lane(ID, LaneID) :- is_vehicle(ID), in_area(LaneID, ID), area(LaneID, traffic_lane).

% A lane transition occurs when a vehicle was observed
% in one traffic lane in the recent past and is currently
% observed in a different traffic lane.
%
% The lookup set {1, 2} checks the last two previous frames,
% rather than only the immediately preceding one, because object
% detection may be noisy and the vehicle may not be detected in
% the previous frame.
#temp lane_transition(ID) :- vehicle_in_traffic_lane(ID, FromLane) at least 1 in {1, 2}, vehicle_in_traffic_lane(ID, ToLane), FromLane != ToLane.

#temp candidate_anomaly(ID, u_turn, "A vehicle may be performing a u-turn", Start, End) :- lane_transition(ID), Start = @now - 10, End = @now + 10.