% Author: Abhishek Kulkarni
% Last Modified: 1 Nov 2016

function [ car ] = action( car, act )
    % Applies action to car. 
    % @param act: straight, leftTurn, rightTurn
    % @param car: Car Object
    %
    % @bug: while turning, the movement should consider speed. For example,
    % if the turn is made with speed of 5, then car may move to north-west
    % cell and step 4 steps forward. (Discuss if this makes sense)

    % Safety against caps-typo
    act = lower(act);

    % Wrapper around actions
    if isequal(act, 'straight')
        car = goStraight(car, car.state.speed);
    elseif isequal(act, 'left')
        car = turnLeft(car);
    elseif isequal(act, 'right')
        car = turnRight(car);
    end

end

function car = goStraight(car, stepSize)
    % We define the left turn as a diagonally left move. 

    if car.state.h == Orientation.north
        car.state.y = car.state.y + stepSize;
        
    elseif car.state.h == Orientation.west
        car.state.x = car.state.x - stepSize;
        
    elseif car.state.h == Orientation.south
        car.state.y = car.state.y - stepSize;

    elseif car.state.h == Orientation.east
        car.state.x = car.state.x + stepSize;
            
    end
    
end

function car = turnLeft(car)
% We define the left turn as a diagonally left move. 

    if car.state.h == Orientation.north
        car.state.x = car.state.x - 1;
        car.state.y = car.state.y + 1;
        car.state.h = Orientation.west;
        
    elseif car.state.h == Orientation.west
        car.state.x = car.state.x - 1;
        car.state.y = car.state.y - 1;
        car.state.h = Orientation.south;
        
    elseif car.state.h == Orientation.south
        car.state.x = car.state.x + 1;
        car.state.y = car.state.y - 1;
        car.state.h = Orientation.east;
        
    elseif car.state.h == Orientation.east
        car.state.x = car.state.x + 1;
        car.state.y = car.state.y + 1;
        car.state.h = Orientation.north;        
    end
    
end

function car = turnRight(car)
% We define the left turn as a diagonally left move. 

    if car.state.h == Orientation.north
        car.state.x = car.state.x + 1;
        car.state.y = car.state.y + 1;
        car.state.h = Orientation.east;
        
    elseif car.state.h == Orientation.west
        car.state.x = car.state.x - 1;
        car.state.y = car.state.y + 1;
        car.state.h = Orientation.north;
        
    elseif car.state.h == Orientation.south
        car.state.x = car.state.x - 1;
        car.state.y = car.state.y - 1;
        car.state.h = Orientation.west;
        
    elseif car.state.h == Orientation.east
        car.state.x = car.state.x + 1;
        car.state.y = car.state.y - 1;
        car.state.h = Orientation.south;        
    end
    
end
