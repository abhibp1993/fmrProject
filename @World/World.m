% this is a class used for generate a world.
% xLength: the number of cell in the horizontal direction.
% yLength: the number of cell in the vertical direction.
% objects: vecotr of objects,
classdef World
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
		
		function this = setX(this, X)
			x = size(this.objects,1);
			y = size(this.objects,2);
			if x<X
				this.objects = [this.objects;cell(X-x, y)];
			elseif x>X
				this.objects = this.objects(1:X,:);
			else
			end
		end
		
		function this = addObject(this, object)
			%% TODO: check to make sure x and y are within bounds!!!
            if object.state.x<1 || object.state.y<1 || object.state.x>this.xLength || object.state.y>this.yLength
                disp('error. object to be added is out of bounds.');
                return;
            end
            if isempty(this.objects{object.state.x,object.state.y}) 
                this.objects(object.state.x,object.state.y) = {object};
            else
                this.objects(object.state.x,object.state.y) = {{this.objects{object.state.x, object.state.y} , object}};
            end
		end
		
		function W = stepWorld(W)
			for i = 1:length(W.objects)
				W.objects(i).step();
			end
		end
		
		function result = checkAP(this, x,y,AP)
			result = AP(this.object{x,y});
		end
	end
	methods
		plotWorld(obj)
	end
end