using CSV
using DataFrames

path = abspath("Datasets\\threshold_distributions_fairness")

if !isfile(joinpath(path, "df_epsilon_fairness.csv")) || !isfile(joinpath(path, "df_epsilon_crit_fairness.csv"))
    error("df of epsilon fairness not found, please run python import first to generate it!")
end

df_epsilon = DataFrame(CSV.File(joinpath(path, "df_epsilon_fairness.csv")))
df_epsilon_crit = DataFrame(CSV.File(joinpath(path, "df_epsilon_crit_fairness.csv")))

Networks = unique(df_epsilon_crit.network)
test_df = @view df_epsilon[df_epsilon.ds .== "test", :]

df_bounds = DataFrame(network = [network for network in Networks for _ in unique(test_df[test_df.network .== network, :].image)],
                      image = [image for network in Networks for image in unique(test_df[test_df.network .== network, :].image)],
                      lower_bound = 0.0,
                      upper_bound = 0.4)


println("Getting bounds...")

for network in Networks
    # get relevant data for that network
    df_epsilon_network = test_df[test_df.network .== network, [:epsilon,:result,:image,:runtime]]

    view_df_bounds = @view df_bounds[df_bounds.network .== network, :]

    # get upper and lower bound for each image
    for img in unique(df_epsilon_network.image)
        epsilon_for_image = @view df_epsilon_network[df_epsilon_network.image .== img, :]
        unsat_epsilons = epsilon_for_image[epsilon_for_image.result .== "unsat", :].epsilon
        sat_epsilons = epsilon_for_image[epsilon_for_image.result .== "sat", :].epsilon

        row_index = findfirst(view_df_bounds.image .== img)

        if !isempty(unsat_epsilons)
            view_df_bounds[row_index, :lower_bound] = maximum(unsat_epsilons)
        else
            println("empty, unsat", ' ', img, ' ', network)
        end

        if !isempty(sat_epsilons)
            view_df_bounds[row_index, :upper_bound] = minimum(sat_epsilons)
        else
            println("empty, sat", ' ', img, ' ', network)
        end
    end
end