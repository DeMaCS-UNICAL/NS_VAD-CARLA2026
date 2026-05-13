% =========================
% Jaywalking example
% =========================
%
% This is a simplified example of jaywalking detection. The legal definition
% of jaywalking may vary depending on the country, state, or local regulation.
% In this example, jaywalking is modeled as a pedestrian crossing or standing
% in a traffic lane while not being inside a crosswalk area.
%


% =========================
% Area membership
% =========================

% Note: the crosswalk area is not present in the current example input, but it
% is kept in the model to show how overlapping semantic areas can be represented.
% For instance, a crosswalk may geometrically overlap with a traffic lane, and
% the reasoning layer can distinguish between a pedestrian legally crossing
% inside the crosswalk and a pedestrian located in the traffic lane outside it.
#temp in_crosswalk(ObjectID) :- in_area(AreaID, ObjectID), area(AreaID, crosswalk).
#temp in_traffic_lane(ObjectID) :- in_area(AreaID, ObjectID), area(AreaID, traffic_lane).

% =========================
% Jaywalking condition
% =========================

#temp pedestrian(ID) :- object(ID, person, _, _).
#temp pedestrian_outside_crosswalk(ID) :- pedestrian(ID), in_traffic_lane(ID), not in_crosswalk(ID).
#temp anomaly(ID, jaywalking, @now) :- pedestrian_outside_crosswalk(ID).