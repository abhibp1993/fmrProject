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
        car = goStraight(car, car.speed);
    elseif isequal(act, 'left')
        car = turnLeft(car);
    elseif isequal(act, 'right')
        car = turnRight(car);
    end

end

function car = goStraight(car, stepSize)
    % We define the left turn as a diagonally left move. 

    if car.h == Orientation.north
        car.y = car.y + stepSize;
        
    elseif car.h == Orientation.west
        car.x = car.x - stepSize;
        
    elseif car.h == Orientation.south
        car.y = car.y - stepSize;

    elseif car.h == Orientation.east
        car.x = car.x + stepSize;
            
    end
    
end

function car = turnLeft(car)
% We define the left turn as a diagonally left move. 

    if car.h == Orientation.north
        car.x = car.x - 1;
        car.y = car.y + 1;
        car.h = Orientation.west;
        
    elseif car.h == Orientation.west
        car.x = car.x - 1;
        car.y = car.y - 1;
        car.h = Orientation.south;
        
    elseif car.h == Orientation.south
        car.x = car.x + 1;
        car.y = car.y - 1;
        car.h = Orientation.east;
        
    elseif car.h == Orientation.east
        car.x = car.x + 1;
        car.y = car.y + 1;
        car.h = Orientation.north;        
    end
    
end

function car = turnRight(car)
% We define the left turn as a diagonally left move. 

    if car.h == Orientation.north
        car.x = car.x + 1;
        car.y = car.y + 1;
        car.h = Orientation.east;
        
    elseif car.h == Orientation.west
        car.x = car.x - 1;
        car.y = car.y + 1;
        car.h = Orientation.north;
        
    elseif car.h == Orientation.south
        car.x = car.x - 1;
        car.y = car.y - 1;
        car.h = Orientation.west;
        
    elseif car.h == Orientation.east
        car.x = car.x + 1;
        car.y = car.y - 1;
        car.h = Orientation.south;        
    end
    
end
