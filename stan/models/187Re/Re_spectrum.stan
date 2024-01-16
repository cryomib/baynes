vector re_bare_spectrum(vector E){
    real me = 510998;
    return sqrt(E^2 + 2*E*me) * E * (19.95./E + 1 − 6.80e-6 * E +3.05e-9  * E^2 );
}
