% Last modified: 6 Nov 2016
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
		
		function this = removeObject(this, i,j,k)
			if(length(this.objects{i,j}) ==1)
				this.objects{i,j} = [];
			else
				this.objects{i,j} = [];
			end
		end
		
		function this = addObject(this, object)
			%% TODO: check to make sure x and y are within bounds!!!
			if isempty(this.objects{object.state.x,object.state.y})
				this.objects(object.state.x,object.state.y) = {object};
			else
				this.objects(object.state.x,object.state.y) = {{this.objects{object.state.x, object.state.y} , object}};
			end
		end
		
		function W = stepWorld(W)
			nW = W;
			for i = 1:size(W.objects,1)
				for j = 1:size(W.objects,2)
                   if length(W.objects{i,j}) > 1
                       for k = 1:length(W.objects{i,j})
                            nW.removeObject(i,j,k);
                            W.objects{i,j}{k}.step(1);
                            nW.addObject(W.objects{i,j}{k});
                       end
                   elseif length(W.objects{i,j}) == 1
                       nW = nW.removeObject(i,j,0);
                       obj = W.objects{i,j}.step(1);
                       nW = nW.addObject(obj);
                   end
				end
			end
			W = nW;
		end
			
			function result = checkAP(this, x,y,AP)
				result = AP(this.object{x,y});
			end
		end
		methods
			plotWorld(obj)
		end
	end