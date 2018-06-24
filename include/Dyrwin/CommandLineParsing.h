//
// Created by Elias Fernandez on 22/06/2018.
//

#ifndef DYRWIN_COMMANDLINEPARSING_H
#define DYRWIN_COMMANDLINEPARSING_H

#include <vector>
#include <iostream>

#include <boost/program_options.hpp>

/*
 * How to use.
 *
 * First you declare the option variables, then you create an Options object,
 * and add the options using the provided functions.
 *
 * Finally, you parse the command line.
 *
 *    size_t nodes;
 *    unsigned experiments;
 *    unsigned timesteps;
 *    std::string filename;
 *    Options options;
 *    options.push_back(makeRequiredOption("nodes,n", &nodes, "set the experiment's nodes number"));
 *    options.push_back(makeDefaultedOption("experiments,e", &experiments, "set the number of experiments", 1000u));
 *    options.push_back(makeDefaultedOption("timesteps,t", &timesteps, "set the timesteps per experiment", 500u));
 *    options.push_back(makeRequiredOption("output,o", &filename, "set the final output file"));
 *
 *    if (!parseCommandLine(argc, argv, options))
 *        return 1;
 */

namespace po = boost::program_options;

using Option = std::tuple<std::string, const po::value_semantic *, std::string>;
using Options = std::vector<Option>;

template <typename T>
Option makeRequiredOption(std::string opt, T* value, std::string description) {
    return std::make_tuple(opt, po::value(value)->required(), description);
}

template <typename T>
Option makeDefaultedOption(std::string opt, T* value, std::string description, T def) {
    return std::make_tuple(opt, po::value(value)->default_value(def), description);
    // ->implicit(__) specified no value
}

inline bool parseCommandLine(int argc, char** argv, const Options & options) {
    try {
        po::options_description desc("Options");

        auto inserter = desc.add_options();
        inserter("help,h", "produce help message");

        for (const auto & option : options)
            inserter(std::get<0>(option).c_str(), std::get<1>(option), std::get<2>(option).c_str());

        po::positional_options_description p;
        // p.add("port", -1);

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).
                options(desc).positional(p).run(), vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return false;
        }

        po::notify(vm);
    }
    catch(const boost::program_options::error& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return false;
    }

    return true;
}

#endif //DYRWIN_COMMANDLINEPARSING_H
