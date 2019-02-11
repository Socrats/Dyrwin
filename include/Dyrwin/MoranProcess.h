//
// Created by Elias Fernandez on 15/03/2018.
//

#ifndef DYRWIN_MORANPROCESS_H
#define DYRWIN_MORANPROCESS_H

template<class P, class S>
class MoranProcess {
    /**
     * This class implements the Moran process with pairwise comparison
     */
public:
    MoranProcess(T *population, S &selection, double beta) : selection(selection), _beta(beta),
                                                             _population(population) {};

    ~MoranProcess() = default;

    void next();

    Number& operator++() { // prefix ++
        Number result(*this);
        return next();
    }

    S &selection;

private:
    double _beta;
    T *_population;
};


#endif //DYRWIN_MORANPROCESS_H
