% =========================
% Static knowledge
% =========================

allowed_direction(area0, south). 
allowed_direction(area1, north).

vehicle(car). vehicle(truck). vehicle(motorcycle). vehicle(bus).

% =========================
% Vehicle detection
% =========================
#temp is_vehicle(ID) :- object(ID, Class, _, _), vehicle(Class).
#temp vehicle_in_area(ID, AreaID) :- is_vehicle(ID), in_area(AreaID, ID).

% =========================
% Movement estimation
% =========================

% Computes the vertical displacement of a vehicle by comparing
% its current Y coordinate with a previous Y coordinate observed
% within the last time point.
#temp delta_y(ID, DY) :- object(ID, Class, _, Y1) at least 1 in {1}, object(ID, Class, _, Y2), vehicle(Class), DY = Y2 - Y1.

#temp direction(ID, north) :- delta_y(ID, DY), DY < 0.
#temp direction(ID, south) :- delta_y(ID, DY), DY > 0.

% Wrong-way driving
anomaly(ID, wrong_way, @now) :- vehicle_in_area(ID, AreaID), direction(ID, Dir), allowed_direction(AreaID, AllowedDir), Dir != AllowedDir.
