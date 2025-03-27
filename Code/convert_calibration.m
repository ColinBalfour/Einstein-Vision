function convert_calibration(input_file, output_file)
    % Load the original calibration .mat file
    data = load(input_file);  % This gives you a struct with fields 'back','front','left','right'

    var_names = fieldnames(data);
    disp("Top-level variables in the file:")
    disp(var_names)

    % For each top-level variable (e.g. data.back, data.front, etc.)
    for i = 1:length(var_names)
        var_name = var_names{i};
        disp("Examining variable: " + var_name)

        obj = data.(var_name);
        % If it's a cameraParameters/fisheyeParameters object, we convert it:
        try
            disp("Attempting toStruct on " + var_name)
            data.(var_name) = toStruct(obj);
        catch ME
            % If toStruct() fails (maybe because it's fisheyeParameters and your version
            % doesn't have toStruct, or it's a struct already), try struct(...) as a fallback:
            disp("toStruct failed, trying struct(...). Error was:")
            disp(ME.message)
            try
                data.(var_name) = struct(obj);
            catch ME2
                disp("struct(...) also failed. Keeping as-is. Error was:")
                disp(ME2.message)
            end
        end

        % Now data.(var_name) is hopefully a plain struct
    end

    % Save all top-level variables back into a new file, in standard -v7 format
    disp("Saving converted data to " + output_file)
    save(output_file, '-struct', 'data', '-v7');
end
