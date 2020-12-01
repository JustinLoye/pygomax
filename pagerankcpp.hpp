#ifndef PAGERANKCPP_HPP
#define PAGERANKCPP_HPP
#include <vector>
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::MatrixXd;

template <typename T>
int find_arg_in_vector(const std::vector<T> &tosearch, T value)
{
/*
Find return argument of the first encountered value
*/
  for(int i=0; i<tosearch.size(); i++)
  {
    if(tosearch[i]==value)
    {
      return i;
    }
  }
  return -1;
}

//information stored about nodes
struct t_page
{
    int nbOut;
    double sumWeight;
    std::pair<std::vector<int>, std::vector<double>> outLinks;
};
typedef struct t_page t_page;

class Gmatrix
{

private:

  // Basic Google Matrix attributes
  int m_N, m_L;
  double m_alpha; 
  std::vector<t_page> m_Smatrix;

  // Attributes for regomax
  int m_Nr;
  double m_lambda_c;
  std::vector<int> m_rnodes;
  Eigen::VectorXd m_PsiR, m_PsiL;

  // Intermediary steps for REGOMAX
  Eigen::MatrixXd compute_Grr();
  void Project_rnodes(VectorXd& Psi);
  double compute_PsiR(int MAXITER);
  double compute_PsiL(int MAXITER);
  Eigen::MatrixXd compute_Gpr();
  void Project_Pc(VectorXd& vector);
  void Project_Qc(VectorXd& vector);
  MatrixXd compute_Gqr(int MAXITER);

public:

  Gmatrix(std::vector<int> from, std::vector<int> to, std::vector<double> weight, double alpha);
  Gmatrix(){};

  void add_link_G(int i, int j, double k);
  double get_G(int i, int j) const;
  void power_iteration(Eigen::VectorXd &in, Eigen::VectorXd &out) const;
  
  Eigen::VectorXd compute_PageRank(int MAXITER=150);
  std::vector<Eigen::MatrixXd> compute_Regomax(std::vector<int> rnodes, int MAXITER=150);
};

#endif