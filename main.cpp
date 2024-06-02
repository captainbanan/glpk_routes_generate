#include "stdio.h"
#include "glpk.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <fstream>
#include <time.h>
#include "json.hpp"


#define EPSILON 1e-6

typedef std::vector<std::pair<int, int>> matrix_row;
typedef std::vector<matrix_row> matrix;

nlohmann::json INPUT_CONFIG;

static double haversine(double lat1, double lon1,
                        double lat2, double lon2) {
    // distance between latitudes
    // and longitudes
    double dLat = (lat2 - lat1) *
                  M_PI / 180.0;
    double dLon = (lon2 - lon1) *
                  M_PI / 180.0;

    // convert to radians
    lat1 = (lat1) * M_PI / 180.0;
    lat2 = (lat2) * M_PI / 180.0;

    // apply formulae
    double a = pow(sin(dLat / 2), 2) +
               pow(sin(dLon / 2), 2) *
               cos(lat1) * cos(lat2);
    double rad = 6371;
    double c = 2 * asin(sqrt(a));
    return rad * c;
}

class CSVReader {
private:
    std::string m_filename;
    std::vector<std::string> headers;
    std::vector<std::unordered_map<std::string, std::string>> data;

    void parseHeaders(const std::string &line) {
        std::istringstream ss(line);
        std::string header;
        while (std::getline(ss, header, ',')) {
            headers.push_back(header);
        }
    }

    void parseLine(const std::string &line) {
        std::istringstream ss(line);
        std::string field;
        std::unordered_map<std::string, std::string> row;
        size_t index = 0;

        while (std::getline(ss, field, ',') && index < headers.size()) {
            row[headers[index]] = field;
            ++index;
        }
        data.push_back(row);
    }

public:
    CSVReader(const std::string &filename) : m_filename(filename) {}

    bool readCSV() {
        std::ifstream file(m_filename);
        if (!file.is_open()) {
            std::cerr << "Ошибка при открытии файла: " << m_filename << std::endl;
            return false;
        }

        std::string line;
        // Считывание первой строки для заголовков
        if (std::getline(file, line)) {
            parseHeaders(line);
        } else {
            return false;
        }

        // Считывание остальных данных
        while (std::getline(file, line)) {
            parseLine(line);
        }
        file.close();
        return true;
    }

    void printData() const {
        for (const auto &row: data) {
            for (const auto &col: row) {
                std::cout << col.first << ": " << col.second << "; ";
            }
            std::cout << std::endl;
        }
    }

    std::vector<std::unordered_map<std::string, std::string>> &getData() {
        return this->data;
    }
};


bool AreSame(double a, double b) {
    return std::fabs(a - b) < EPSILON;
}


class ClaimPoint {
public:
    double lat{}, lon{};
    std::string claim_id, point_id;

    ClaimPoint(double lat, double lon, const std::string &claim_id, bool is_source) {
        this->lat = lat;
        this->lon = lon;
        std::string suffix = "source";
        if (not is_source) {
            suffix = "destination";
        }
        this->point_id = claim_id + "_" + suffix;
        this->claim_id = claim_id;
    }

    ClaimPoint() = default;

    ~ClaimPoint() = default;
};

class Claim {
public:
    std::string claim_id;
    ClaimPoint pickup_point, delivery_point;
    double distance{};

    Claim(const std::string &claim_id, double source_lat, double source_lon,
          double destination_lat, double destination_lon) {
        this->claim_id = claim_id;
        this->pickup_point = ClaimPoint(source_lat, source_lon,
                                        claim_id, true);
        this->delivery_point = ClaimPoint(destination_lat, destination_lon,
                                          claim_id, false);

        this->distance = haversine(source_lat, source_lon, destination_lat, destination_lon);
    }

    [[nodiscard]] std::string id() const {
        return this->claim_id;
    }

    Claim() = default;

    ~Claim() = default;
};

class Edge {
private:
    ClaimPoint source, destination;
    double distance;
    std::string edge_id;

public:
    Edge(ClaimPoint &source, ClaimPoint &destination) : source(source), destination(destination) {
        this->distance = haversine(source.lat, source.lon, destination.lat, destination.lon);
        this->edge_id = make_edge_id(source.point_id, destination.point_id);
    }

    static std::string make_edge_id(std::string first_point_id, std::string second_point_id) {
        return first_point_id + "_" + second_point_id;
    }

    [[nodiscard]] std::string id() const {
        return this->edge_id;
    }

    [[nodiscard]] std::string get_source_id() const {
        return this->source.point_id;
    }

    [[nodiscard]] std::string get_destination_id() const {
        return this->destination.point_id;
    }

    [[nodiscard]] double get_distance() const {
        return this->distance;
    }

    Edge() = default;

    ~Edge() = default;
};

enum constraint_type {
    LEQ = 0,
    EQ = 1
};

std::vector<double> INIT_SOLUTION;

void callback(glp_tree *tree, void *info) {
    switch (glp_ios_reason(tree)) {
        case GLP_IHEUR:
            glp_ios_heur_sol(tree, &INIT_SOLUTION[0]);
            break;
        default:
            break;
    }
}

class IlpWrapper {
    /*
     * A1 @ x <= b1
     * A2 @ x = b2
     * c @ x -> max
    */
private:
    matrix A1, A2;
    std::vector<int> b1, b2, variables_types;
    std::vector<double> c;
    std::unordered_map<std::string, int> var_name_to_idx;
    glp_prob *ILP;
    std::unordered_map<std::string, double> var_name_to_val;
    double optimum;

public:
    IlpWrapper() {
        this->A1 = matrix();
        this->A2 = matrix();
        this->variables_types = std::vector<int>();
        this->b1 = std::vector<int>();
        this->b2 = std::vector<int>();
        this->c = std::vector<double>();
        this->var_name_to_idx = {};
        this->var_name_to_val = {};
        this->optimum = 0;
    }

    double get_optimum() {
        return this->optimum;
    }

    void add_variable(std::string variable_id, double objective, int variable_type) {
        int variables_count = this->c.size();
        var_name_to_idx.insert(std::make_pair(variable_id, variables_count));
        this->c.push_back(objective);
        this->variables_types.push_back(variable_type);
    }

    void add_constraint(std::vector<std::string> variables_names,
                        std::vector<int> coeffs, int b, constraint_type type) {
        std::unordered_set<std::string> names;
        for (auto const &name: variables_names) {
            names.insert(name);
        }
        if (names.size() != variables_names.size()) {
            std::cout << "two equal names\n";
        }

        matrix_row row = matrix_row();
        for (int i = 0; i < variables_names.size(); i++) {
            int variable_index = this->var_name_to_idx[variables_names[i]];
            row.emplace_back(variable_index, coeffs[i]);
        }
        if (type == LEQ) {
            this->A1.push_back(row);
            this->b1.push_back(b);
        } else {
            this->A2.push_back(row);
            this->b2.push_back(b);
        }
    }

    void create_ilp() {
        this->ILP = glp_create_prob();
        int vars_count = this->c.size();
        int rows_count = this->A1.size() + this->A2.size();
        glp_add_cols(this->ILP, vars_count);
        glp_add_rows(this->ILP, rows_count);
        for (int i = 0; i < this->c.size(); i++) {
            glp_set_col_kind(this->ILP, i + 1, this->variables_types[i]);
            glp_set_obj_coef(this->ILP, i + 1, this->c[i]);
            glp_set_col_bnds(this->ILP, i + 1, GLP_LO, 0, NULL);
        }

        for (int i = 0; i < this->A1.size(); i++) {
            auto row = this->A1[i];
            std::vector<int> idxs(1);
            std::vector<double> coeffs(1);
            for (auto &pair: row) {
                idxs.push_back(pair.first + 1);
                coeffs.push_back(pair.second);
            }
            glp_set_mat_row(this->ILP, i + 1, row.size(), &idxs[0], &coeffs[0]);
            glp_set_row_bnds(this->ILP, i + 1, GLP_UP, NULL, this->b1[i]);
        }

        for (int i = 0; i < this->A2.size(); i++) {
            auto row = this->A2[i];
            std::vector<int> idxs(1);
            std::vector<double> coeffs(1);
            for (auto &pair: row) {
                idxs.push_back(pair.first + 1);
                coeffs.push_back(pair.second);
            }
            glp_set_mat_row(this->ILP, i + 1 + this->A1.size(), row.size(), &idxs[0], &coeffs[0]);
            glp_set_row_bnds(this->ILP, i + 1 + this->A1.size(), GLP_FX, this->b2[i], this->b2[i]);
        }

        glp_set_obj_dir(this->ILP, GLP_MAX);

    }

    void set_initial_solution(std::vector<std::pair<std::string, double>> &var_name_to_value) {
        INIT_SOLUTION.resize(this->var_name_to_idx.size() + 1);
        std::fill(INIT_SOLUTION.begin(), INIT_SOLUTION.end(), 0);
        for (auto const &[var_name, value]: var_name_to_value) {
            int idx = this->var_name_to_idx[var_name];
            INIT_SOLUTION[idx + 1] = value;
        }
    }

    int find_opt() {
        glp_iocp iocp;
        glp_init_iocp(&iocp);

        iocp.br_tech = GLP_BR_DTH; /* most fractional variable */
        iocp.bt_tech = GLP_BT_BFS; /* best local bound */
//        iocp.sr_heur = GLP_ON; /* disable simple rounding heuristic */
//        iocp.gmi_cuts = GLP_ON; /* enable Gomory cuts */
        iocp.pp_tech = GLP_PP_ROOT;
//        iocp.fp_heur = GLP_ON;

//        iocp.ps_heur = GLP_ON;

//        iocp.binarize = GLP_ON;

        iocp.mir_cuts = GLP_ON;
//        iocp.cov_cuts = GLP_ON;
        iocp.msg_lev = GLP_MSG_ON;

//        iocp.presolve = GLP_ON;

        iocp.tm_lim = 15000;

        glp_smcp parm;
        glp_init_smcp(&parm);

        parm.msg_lev = GLP_MSG_ON;
        parm.tm_lim = 15000;

        if (INPUT_CONFIG["use_initial_solution"]) {
            iocp.cb_func = callback;
            glp_simplex(this->ILP, &parm);
        } else {
            iocp.presolve = GLP_ON;
        }

        int result = glp_intopt(this->ILP, &iocp);

        for (auto const &[var_name, index]: this->var_name_to_idx) {
            double optimization_result = glp_mip_col_val(this->ILP, index + 1);
            double objective = this->c[index];
            this->optimum += objective * optimization_result;
            this->var_name_to_val.insert(std::make_pair(var_name, optimization_result));
        }



        return result;
    }

    std::unordered_map<std::string, double> get_result() {
        return this->var_name_to_val;
    }

    ~IlpWrapper() = default;


};

class Graph {
private:
//    std::unordered_map<std::string, int> edge_id_to_variable_idx;
    std::unordered_map<std::string, std::unordered_set<std::string>> point_id_to_outgoing_edges, point_id_to_ingoing_edges;
    std::unordered_map<std::string, ClaimPoint> point_id_to_point;
    std::unordered_map<std::string, Edge> edge_id_to_edge;
    std::unordered_map<std::string, Claim> claim_id_to_claim;

    int route_len;
    IlpWrapper ilp;

public:
    std::unordered_map<std::string, double> edge_id_to_result;
    double sum_route_len = 0;
    double claim_sum_len = 0;


    Graph() {
        this->edge_id_to_edge = {};
        this->point_id_to_point = {};
        this->point_id_to_outgoing_edges = {};
        this->point_id_to_ingoing_edges = {};
        this->claim_id_to_claim = {};
        this->edge_id_to_result = {};
        this->ilp = IlpWrapper();
        this->route_len = -1;
    }

    void add_vertex(ClaimPoint &point) {
        this->point_id_to_point.insert(std::make_pair(point.point_id, point));
    }

    void add_claim(Claim &claim) {
        this->claim_id_to_claim.insert(std::make_pair(claim.claim_id, claim));
        this->add_vertex(claim.pickup_point);
        this->add_vertex(claim.delivery_point);
    }

    void add_edge(Edge &edge) {
        if (edge_id_to_edge.contains(edge.id())) {
            std::cout << "edge already in graph";
        }
        this->edge_id_to_edge.insert(std::make_pair(edge.id(), edge));
        auto source_id = edge.get_source_id();
        auto destination_id = edge.get_destination_id();
        this->point_id_to_outgoing_edges[source_id].insert(edge.id());
        this->point_id_to_ingoing_edges[destination_id].insert(edge.id());
    }

    static std::string var_name_for_ith_bit(std::string old_var_name, int bit_num) {
        return old_var_name + "_" + std::to_string(bit_num);
    }

    Edge &get_edge(std::string edge_id) {
        return this->edge_id_to_edge[edge_id];
    }

    Claim &get_claim_by_point_id(std::string point_id) {
        std::string claim_id = this->point_id_to_point[point_id].claim_id;
        return this->claim_id_to_claim[claim_id];
    }

    Claim &get_claim_by_claim_id(std::string claim_id) {
        return this->claim_id_to_claim[claim_id];
    }

    void create_linear_program() {
        std::vector<int> inequalities_to_use_vec = INPUT_CONFIG["inequalities_to_use"];
        auto inequalities_to_use = std::unordered_set(inequalities_to_use_vec.begin(),
                                                      inequalities_to_use_vec.end());
        int batch_size = INPUT_CONFIG["batch_size"];

        this->route_len = batch_size * 2 -1;

        // добавляем k битов на каждое ребро
        for (auto const &[edge_id, edge]: this->edge_id_to_edge) {
            std::string claim_id = this->point_id_to_point[edge.get_source_id()].claim_id;
            Claim claim = this->claim_id_to_claim[claim_id];
            double claim_distance = 0;
            if (claim.pickup_point.point_id == edge.get_source_id()) {
                claim_distance = claim.distance;
            }

            for (int i = 0; i < route_len; i++) {
                std::string var_name = this->var_name_for_ith_bit(edge.id(), i);
                ilp.add_variable(var_name, claim_distance - edge.get_distance(), GLP_BV);
            }
        }

        // добавляем мнимые начальные и конечные ребра
        for (auto const &[point_id, point]: this->point_id_to_point) {
            std::string var_name = point_id + "_start";
            ilp.add_variable(var_name, 0, GLP_BV);

            var_name = point_id + "_end";
            ilp.add_variable(var_name, 0, GLP_BV);
        }



        if (inequalities_to_use.contains(1)) {
            // по каждому биту сумма по всем ребрам равна 1
            for (int i = 0; i < route_len; i++) {
                std::vector<std::string> var_names;
                for (auto const &[edge_id, edge]: this->edge_id_to_edge) {
                    std::string var_name = Graph::var_name_for_ith_bit(edge.id(), i);
                    var_names.push_back(var_name);
                }
                ilp.add_constraint(var_names, std::vector<int>(edge_id_to_edge.size(), 1), 1, EQ);
            }
        }

        if (inequalities_to_use.contains(2)) {
            // по каждому биту сумма по всем ребрам не больше 1
            for (int i = 0; i < route_len; i++) {
                std::vector<std::string> var_names;
                for (auto const &[edge_id, edge]: this->edge_id_to_edge) {
                    std::string var_name = Graph::var_name_for_ith_bit(edge.id(), i);
                    var_names.push_back(var_name);
                }
                ilp.add_constraint(var_names, std::vector<int>(edge_id_to_edge.size(), 1), 1, LEQ);
            }
        }

        if (inequalities_to_use.contains(3)) {
            // добавляем огранчение что в каждом ребре не больше одного бита
            for (auto const &[edge_id, edge]: this->edge_id_to_edge) {
                std::vector<std::string> var_names(route_len);
                double claim_distance = 0;

                for (int i = 0; i < route_len; i++) {
                    std::string var_name = this->var_name_for_ith_bit(edge.id(), i);
                    var_names[i] = var_name;
                }
                ilp.add_constraint(var_names, std::vector<int>(route_len, 1), 1, LEQ);
            }
        }

        if (inequalities_to_use.contains(4)) {
            // для каждого бита сумма исходящих ребер должна быть равна сумме входящих для предыдущего бита
            for (auto const &[point_id, point]: this->point_id_to_point) {
                for (int i = 1; i < route_len; i++) {
                    std::vector<std::string> var_names;
                    std::vector<int> coeffs;
                    for (auto const &edge_id: this->point_id_to_outgoing_edges[point_id]) {
                        std::string var_name = Graph::var_name_for_ith_bit(edge_id, i);
                        var_names.push_back(var_name);
                        coeffs.push_back(1);
                    }

                    for (auto const &edge_id: this->point_id_to_ingoing_edges[point_id]) {
                        std::string var_name = Graph::var_name_for_ith_bit(edge_id, i - 1);
                        var_names.push_back(var_name);
                        coeffs.push_back(-1);
                    }
                    ilp.add_constraint(var_names, coeffs, 0, EQ);
                }
            }

            for (auto const &[point_id, point]: this->point_id_to_point) {
                std::vector<std::string> var_names;
                std::vector<int> coeffs;

                std::string start_id = point_id + "_start";
                var_names.push_back(start_id);
                coeffs.push_back(1);
                for (auto const &edge_id: this->point_id_to_outgoing_edges[point_id]) {
                    std::string var_name = Graph::var_name_for_ith_bit(edge_id, 0);
                    var_names.push_back(var_name);
                    coeffs.push_back(-1);
                }
                ilp.add_constraint(var_names, coeffs, 0, EQ);

                var_names.clear();
                coeffs.clear();

                std::string end_id = point_id + "_end";
                var_names.push_back(end_id);
                coeffs.push_back(1);
                for (auto const &edge_id: this->point_id_to_ingoing_edges[point_id]) {
                    std::string var_name = Graph::var_name_for_ith_bit(edge_id, route_len - 1);
                    var_names.push_back(var_name);
                    coeffs.push_back(-1);
                }
                ilp.add_constraint(var_names, coeffs, 0, EQ);
            }
        }

        if (inequalities_to_use.contains(5)) {
            // из pickup точки одной заявки выходит столько же ребер сколько входит в delivery точку этой же заявки
            for (auto const &[claim_id, claim]: this->claim_id_to_claim) {
                const ClaimPoint &pickup_point = claim.pickup_point;
                const ClaimPoint &delivery_point = claim.delivery_point;
                std::vector<std::string> var_names;
                std::vector<int> coeffs;

                for (auto const &edge_id: this->point_id_to_outgoing_edges[pickup_point.point_id]) {
                    if (this->edge_id_to_edge[edge_id].get_destination_id() == delivery_point.point_id) {
                        continue;
                    }
                    for (int i = 0; i < route_len; i++) {
                        std::string var_name = Graph::var_name_for_ith_bit(edge_id, i);
                        var_names.push_back(var_name);
                        coeffs.push_back(1);
                    }
                }

                for (auto const &edge_id: this->point_id_to_ingoing_edges[delivery_point.point_id]) {
                    if (this->edge_id_to_edge[edge_id].get_source_id() == pickup_point.point_id) {
                        continue;
                    }
                    for (int i = 0; i < route_len; i++) {
                        std::string var_name = Graph::var_name_for_ith_bit(edge_id, i);
                        var_names.push_back(var_name);
                        coeffs.push_back(-1);
                    }
                }

                ilp.add_constraint(var_names, coeffs, 0, EQ);
            }
        }

        if (inequalities_to_use.contains(6)) {
            // в каждой заявке первая точка имеет номер до второй
            for (auto const &[claim_id, claim]: this->claim_id_to_claim) {
                const ClaimPoint &pickup_point = claim.pickup_point;
                const ClaimPoint &delivery_point = claim.delivery_point;

                std::vector<std::string> var_names;
                std::vector<int> coeffs;

                for (int i = 0; i < route_len; i++) {
                    for (auto const &edge_id: this->point_id_to_ingoing_edges[pickup_point.point_id]) {
                        std::string var_name = Graph::var_name_for_ith_bit(edge_id, i);
                        var_names.push_back(var_name);
                        coeffs.push_back(std::pow(2, i));
                    }

                    for (auto const &edge_id: this->point_id_to_ingoing_edges[delivery_point.point_id]) {
                        std::string var_name = Graph::var_name_for_ith_bit(edge_id, i);
                        var_names.push_back(var_name);
                        coeffs.push_back(-std::pow(2, i));
                    }
                }

                ilp.add_constraint(var_names, coeffs, 0, LEQ);
            }
        }

        if (inequalities_to_use.contains(7)) {
            // стартовые (конечные) ребра в сумме ровно 1
            std::vector<std::string> var_names;
            std::vector<int> coeffs;
            for (auto const &[point_id, point]: this->point_id_to_point){
                std::string var_name = point_id + "_start";
                var_names.push_back(var_name);
                coeffs.push_back(1);
            }
            ilp.add_constraint(var_names, coeffs, 1, EQ);

            var_names.clear();
            coeffs.clear();

            for (auto const &[point_id, point]: this->point_id_to_point){
                std::string var_name = point_id + "_end";
                var_names.push_back(var_name);
                coeffs.push_back(1);
            }
            ilp.add_constraint(var_names, coeffs, 1, EQ);
        }

        if (inequalities_to_use.contains(8)) {
            // для каждой вершины сумма входящий битов на соответствующие степени двойки в два раза меньше исходящих
            for (auto const &[point_id, point]: this->point_id_to_point) {
                std::vector<std::string> var_names;
                std::vector<int> coeffs;
                std::string start_id = point_id + "_start";
                var_names.push_back(start_id);
                coeffs.push_back(-1);

                std::string end_id = point_id + "_end";
                var_names.push_back(end_id);
                coeffs.push_back(std::pow(2, route_len));

                for (int i = 0; i < route_len; i++) {
                    for (auto const &edge_id: this->point_id_to_outgoing_edges[point_id]) {
                        std::string var_name = Graph::var_name_for_ith_bit(edge_id, i);
                        var_names.push_back(var_name);
                        coeffs.push_back(std::pow(2, i));
                    }

                    for (auto const &edge_id: this->point_id_to_ingoing_edges[point_id]) {
                        std::string var_name = Graph::var_name_for_ith_bit(edge_id, i);
                        var_names.push_back(var_name);
                        coeffs.push_back(-2 * std::pow(2, i));
                    }
                }
                ilp.add_constraint(var_names, coeffs, 0, EQ);
            }
        }

        if (inequalities_to_use.contains(9)) {
            // сумма входящих(выходящих) ребер не больше 1
            for (auto const &[point_id, point]: this->point_id_to_point) {
                std::vector<std::string> var_names;
                std::vector<int> coeffs;

                std::string start_id = point_id + "_start";
                var_names.push_back(start_id);
                coeffs.push_back(1);
                for (int i = 0; i < route_len; i++) {
                    for (auto const &edge_id: this->point_id_to_ingoing_edges[point_id]) {
                        std::string var_name = Graph::var_name_for_ith_bit(edge_id, i);
                        var_names.push_back(var_name);
                        coeffs.push_back(1);
                    }
                }
                ilp.add_constraint(var_names, coeffs, 1, LEQ);
                var_names.clear();
                coeffs.clear();

                std::string end_id = point_id + "_end";
                var_names.push_back(end_id);
                coeffs.push_back(1);
                for (int i = 0; i < route_len; i++) {
                    for (auto const &edge_id: this->point_id_to_outgoing_edges[point_id]) {
                        std::string var_name = Graph::var_name_for_ith_bit(edge_id, i);
                        var_names.push_back(var_name);
                        coeffs.push_back(1);
                    }
                }
                ilp.add_constraint(var_names, coeffs, 1, LEQ);
            }
        }

        if (inequalities_to_use.contains(10)) {
            // сумма всех битов, входящих и выходящих из вершины не больше 2
            for (auto const &[point_id, point]: this->point_id_to_point) {
                std::vector<std::string> var_names;
                std::vector<int> coeffs;
                for (auto const &edge_id: this->point_id_to_ingoing_edges[point_id]) {
                    for (int i = 0; i < route_len; i++) {
                        std::string var_name = Graph::var_name_for_ith_bit(edge_id, i);
                        coeffs.push_back(1);
                        var_names.push_back(var_name);
                    }
                }

                for (auto const &edge_id: this->point_id_to_outgoing_edges[point_id]) {
                    for (int i = 0; i < route_len; i++) {
                        std::string var_name = Graph::var_name_for_ith_bit(edge_id, i);
                        var_names.push_back(var_name);
                        coeffs.push_back(1);
                    }
                }

                std::string end_id = point_id + "_end";
                var_names.push_back(end_id);
                coeffs.push_back(1);

                std::string start_id = point_id + "_start";
                var_names.push_back(start_id);
                coeffs.push_back(1);

                ilp.add_constraint(var_names, coeffs, 2, LEQ);
            }
        }

        if (inequalities_to_use.contains(11)) {
            // из delivery точки одной заявки выходит не больше ребер чем из pickup точки этой же заявки
            for (auto const &[claim_id, claim]: this->claim_id_to_claim) {
                const ClaimPoint &pickup_point = claim.pickup_point;
                const ClaimPoint &delivery_point = claim.delivery_point;
                std::vector<std::string> var_names;
                std::vector<int> coeffs;

                for (auto const &edge_id: this->point_id_to_outgoing_edges[pickup_point.point_id]) {
                    for (int i = 0; i < route_len; i++) {
                        std::string var_name = Graph::var_name_for_ith_bit(edge_id, i);
                        var_names.push_back(var_name);
                        coeffs.push_back(-1);
                    }
                }

                for (auto const &edge_id: this->point_id_to_outgoing_edges[delivery_point.point_id]) {
                    for (int i = 0; i < route_len; i++) {
                        std::string var_name = Graph::var_name_for_ith_bit(edge_id, i);
                        var_names.push_back(var_name);
                        coeffs.push_back(1);
                    }
                }

                std::string end_id = delivery_point.point_id + "_end";
                var_names.push_back(end_id);
                coeffs.push_back(-1);

                ilp.add_constraint(var_names, coeffs, 0, LEQ);
            }
        }
    }

    void set_initial_route() {
        std::vector<std::pair<std::string, double>> var_name_to_value;
        int edge_num = 0;
        std::string prev_point_id = "";
        std::vector<std::string> point_ids;
        for (auto const &[claim_id, claim]: this->claim_id_to_claim) {
            auto source = claim.pickup_point;
            auto destination = claim.delivery_point;
            if (prev_point_id != "") {
                std::string edge_id = Edge::make_edge_id(prev_point_id, source.point_id);
                std::string var_name = Graph::var_name_for_ith_bit(edge_id, edge_num);
                var_name_to_value.emplace_back(var_name, 1);
                edge_num++;
            }
            std::string edge_id = Edge::make_edge_id(source.point_id, destination.point_id);
            point_ids.push_back(source.point_id);
            point_ids.push_back(destination.point_id);

            std::string var_name = Graph::var_name_for_ith_bit(edge_id, edge_num);
            var_name_to_value.emplace_back(var_name, 1);
            edge_num++;
            prev_point_id = destination.point_id;

            if (edge_num == this->route_len) {
                break;
            }
        }

        std::string first_point = point_ids[0];
        std::string last_point = point_ids[point_ids.size() - 1];

        var_name_to_value.emplace_back(first_point + "_start", 1);
        var_name_to_value.emplace_back(last_point + "_end", 1);

        this->ilp.set_initial_solution(var_name_to_value);
    }

    int optimize() {
        this->set_initial_route();
        this->ilp.create_ilp();
        int result_code = this->ilp.find_opt();
        auto result = this->ilp.get_result();

        int sum_claims = 0;
        for (auto const &[edge_id, edge]: this->edge_id_to_edge) {
            double sum_for_edge = 0;
            for (int i = 0; i < this->route_len; i++) {
                std::string var_name = Graph::var_name_for_ith_bit(edge_id, i);
                if (result[var_name] > 0) {
                    sum_for_edge = i + 1;
                    break;
                }
            }
            if (sum_for_edge > 0) {
                this->sum_route_len += edge.get_distance();
                if (edge.get_source_id().find("source") != std::string::npos) {
                    this->claim_sum_len += claim_id_to_claim[point_id_to_point[edge.get_source_id()].claim_id].distance;
                    sum_claims++;
                }
            }
            this->edge_id_to_result[edge_id] = sum_for_edge;
        }
        return result_code;
    }

    ~Graph() = default;
};


int main(int argc, const char *argv[]) {

    std::ifstream file("/Users/captainbanana/ClionProjects/routes/input_config.json");

    file >> INPUT_CONFIG;

    file.close();

    CSVReader reader("/Users/captainbanana/ClionProjects/routes/claims.csv");
    reader.readCSV();

    Graph graph = Graph();

    std::vector<ClaimPoint> points;

    int claims_count = 0;

    for (auto &row: reader.getData()) {

        Claim claim = Claim(row["claim_uuid"],
                            std::stod(row["source_location_lat"]), std::stod(row["source_location_lon"]),
                            std::stod(row["destination_lat"]), std::stod(row["destination_lon"]));

        graph.add_claim(claim);

        points.push_back(claim.pickup_point);
        points.push_back(claim.delivery_point);
        claims_count++;
    }

    for (int i = 0; i < points.size(); i++) {
        for (int j = i + 1; j < points.size(); j++) {
            auto first_point = points[i];
            auto second_point = points[j];

            auto edge = Edge(first_point, second_point);
            graph.add_edge(edge);

            edge = Edge(second_point, first_point);
            graph.add_edge(edge);
        }
    }
    graph.create_linear_program();

    clock_t tStart = clock();
    int result_code = graph.optimize();
    std::string result_code_str = "optimal";
    if (result_code != 0) {
        if (result_code == GLP_ETMLIM) {
            result_code_str = "time limit";
        } else {
            result_code_str = "error";
        }
    }

    double spent_seconds = ((double) clock() - tStart) / CLOCKS_PER_SEC;
    std::cout << spent_seconds << " seconds taken on optimization\n";

//    for (auto const &[edge_id, result]: graph.edge_id_to_result) {
//        std::cout << edge_id << ": " << result << "\n";
//    }


    std::ofstream output_edges("/Users/captainbanana/CLionProjects/routes/output_edges.csv");

    output_edges << "edge_id,value\n";

    double route_len = 0;

    for (const auto &pair: graph.edge_id_to_result) {
        Edge edge = graph.get_edge(pair.first);
        output_edges << pair.first << "," << pair.second << "\n";
    }
    output_edges.close();


    double sh_economy = (graph.claim_sum_len - graph.sum_route_len) / graph.claim_sum_len;
    std::cout << "claims_sum_distance - route_len: " << graph.claim_sum_len - graph.sum_route_len << "\n";
    std::cout << "sh_economy: " << sh_economy << "\n";

    if (std::isnan(sh_economy) or sh_economy <= 0 ) {
        sh_economy = 0;
    }

    auto output_json = std::ofstream("/Users/captainbanana/CLionProjects/routes/output.json");
    output_json << "{\n";
    output_json << "\"seconds_spent\": " << spent_seconds << ",\n";
    output_json << "\"sh_economy\": " << sh_economy << ",\n";
    output_json << "\"claims_count\": " << claims_count << ",\n";
    output_json << "\"result_code_str\": " << "\"" << result_code_str << "\"" << "\n";
    output_json << "}";
    output_json.close();

    return 0;
}