function [lats, lons] = xy2ll(x, y, latc, lonc)
    % Converts Cartesian coordinates (x, y) back to latitude/longitude.
    % Inputs:
    %   x, y       - Cartesian coordinates (meters).
    %   latc, lonc - Latitude and longitude of the center point (degrees).
    % Outputs:
    %   lats, lons - Vectors of latitudes and longitudes (degrees).

    % Convert center point to radians
    latc = deg2rad(latc);
    lonc = deg2rad(lonc);

    % Earth's radius (in meters)
    earthRad = radiusOfEarth(latc); 

    % Inverse gnomonic projection
    rho = sqrt(x.^2 + y.^2);
    c = atan(rho / earthRad);
    sin_c = sin(c);
    cos_c = cos(c);

    % Calculate latitudes and longitudes
    lats = asin(cos_c * sin(latc) + (y .* sin_c .* cos(latc)) ./ rho);
    lons = lonc + atan2(x .* sin_c, rho .* cos(latc) .* cos_c - y .* sin(latc) .* sin_c);

    % Convert back to degrees
    lats = rad2deg(lats);
    lons = rad2deg(lons);
end

function rad = radiusOfEarth(lat)
    % Approximate Earth's radius as a function of latitude (meters)
    % Reference: WGS-84 ellipsoidal parameters
    a = 6378137; % Equatorial radius
    b = 6356752.314245; % Polar radius
    rad = sqrt(((a^2 * cos(lat)).^2 + (b^2 * sin(lat)).^2) ./ ...
               ((a * cos(lat)).^2 + (b * sin(lat)).^2));
end
