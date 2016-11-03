% this is a class used for generate a world.
% xLength: the number of cell in the horizontal direction.
% yLength: the number of cell in the vertical direction.
% objects: vecotr of objects,
classdef World
	properties
		xLength;
		yLength;
		objects=[];
	end
	
	methods
		function W = World(X, Y)
			W.xLength = X;
			W.yLength = Y;
		end
		
		function plotWorld(World)
			figure;
			plot(obj.xLength,obj.yLength)
		end
		
		function addObject(object)
			objects = [objects;object];
		end
		
		function stepWorld()
			for i = 1:length(objects)
				objects(i).step();
			end
		end
		
		function AP = checkAP(x,y)
			for i = 1:length(objects)
				
				if (sum((object.position) == [x,y]) == 2)
				% todo	
				end
			end
		end
	end
end