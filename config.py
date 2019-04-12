config = {
    # Number of points to be rendered along each dimension
    # i.e. for 10, 10, 10 there will be 1000 points rendered in a cube
    "x_points" : 10,
    "y_points" : 10,
    "z_points" : 10,

    # spawning points:
    # how often to spawn new points
    "spawn_interval" : 10,
    # how many time to spawn points before stopping
    "spawn_batches" : 1,

    # Time configuration:
    # total number of "seconds"
    "T" : 30,
    # Axis dimensions (will be of the form -X/2 to X/2)
    "X" : 400,
    "Y" : 400,
    "Z" : 400,
    # time dilation allows for speeding up or slowing down time
    "time_dilation" : 1/5000,

    # Nt is the total number of time steps per "second, increase to minimize 
    # numerical errors, decrease to minimize runtime
    "Nt" : 25,
}
