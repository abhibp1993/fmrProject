% this is a class used for generate a world.
% xLength: the number of cell in the horizontal direction.
% yLength: the number of cell in the vertical direction.
% objects: vecotr of objects,
classdef World < handle
	properties
		xLength=20;
		yLength=20;
		objects=cell(20,20);
	end
	
	methods
		function this = World(X, Y)
			this.xLength = X;
			this.yLength = Y;
			this.objects=cell(X,Y);
        end
		
        %% TODO: I found the error. You cannot have 2 functions named the same!!!!
		%function plotWorld(obj)
		%	figure;
		%	plot(obj.xLength,obj.yLength)
        %end
		
		function addObject(this, object)
            %% TODO: check to make sure x and y are within bounds!!!
			this.objects(object.state.x,object.state.y) = {[this.objects{object.state.x, object.state.y} , object]};
		end
		
		function stepWorld(W)
			for i = 1:length(W.objects)
				W.objects(i).step();
			end
		end
		
		function result = checkAP(this, x,y,AP)
			result = AP(this.object{x,y});
        end
	end
end