% 6 Nov 2016 by Andy
% setRoad(w, s,e) is a function to generate a road between start point s and
% end point e, in World w.
% The output is the world containing the road.

function w = setRoad(w, s, e)
if (s(1) == e(1))
	id = 1;
	for i = s(2):e(2)
		r = Road(id, s(1),i);
		w = w.addObject(r);
		id = id+1;
	end
elseif(s(2) == e(2))
	id = 1;
	for i = s(1):e(1)
		r = Road(id, s(2),i);
		w = w.addObject(r);
		id = id+1;
	end
else
	disp('start point and end point is not in a line')
	return
end