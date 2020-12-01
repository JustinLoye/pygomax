#include "pagerankcpp.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "pybind11/eigen.h"

using namespace std;
namespace py = pybind11;

Gmatrix::Gmatrix(vector<int> from, vector<int> to, vector<double> weight, double alpha)
{
  m_L = from.size();
  int maxi;
  int i;
  m_N = 0;
  for(i=0; i<m_L; ++i)
  {
    maxi = max(from[i], to[i]);
    if(maxi > m_N){
      m_N = maxi;
    }
  }
  m_N += 1;

  m_Smatrix.resize(m_N);
  for(i=0; i<m_N; ++i)
  {
    m_Smatrix[i].nbOut = 0;
    m_Smatrix[i].sumWeight = 0.0;
  }

  m_alpha = alpha;
  for(i=0; i<m_L; ++i)
  {
    Gmatrix::add_link_G(from[i], to[i], weight[i]);
  }
}

void Gmatrix::add_link_G(int i, int j, double w)
{
  m_Smatrix[i].nbOut += 1;
  m_Smatrix[i].sumWeight += w;
  m_Smatrix[i].outLinks.first.push_back(j);
  m_Smatrix[i].outLinks.second.push_back(w);
}

double Gmatrix::get_G(int i, int j) const
{
  double element;
  int from = find_arg_in_vector(m_Smatrix[j].outLinks.first, i);
  //Deux cas de figure : j est dangling ou non
  if(m_Smatrix[j].nbOut==0)
  {
    element = 1.0/m_N;
  }
  else if(from!=-1)
  {
    element = m_alpha*m_Smatrix[j].outLinks.second[from]/m_Smatrix[j].sumWeight + (1.0-m_alpha)/m_N;
  }
  else
  {
    element = (1.0-m_alpha)/m_N;
  }

  return element;
}

void Gmatrix::power_iteration(VectorXd &v, VectorXd &vbar) const
{
  double const_term;
  int i, j;
  const_term = 0.0;
  for(i=0; i<m_N; i++)
  {
    double factor=1.0;
    if(m_Smatrix[i].nbOut>0)
    {
      factor=1.0-m_alpha;
    }
    const_term += factor*v[i];
  }
  const_term /= m_N;

  vbar = VectorXd::Constant(m_N, const_term);

  for(i=0; i<m_N; i++)
  {
    for(j=0; j<m_Smatrix[i].nbOut; j++)
    {
      vbar[m_Smatrix[i].outLinks.first[j]]+=m_alpha*v[i]*m_Smatrix[i].outLinks.second[j]/m_Smatrix[i].sumWeight;
    }
  }
}

VectorXd Gmatrix::compute_PageRank(int MAXITER)
{
  VectorXd v = VectorXd::Constant(m_N, 1.0/m_N);
  VectorXd vbar = VectorXd::Zero(m_N);
  int niter = 0;
  do
  {
    power_iteration(v, vbar);
    vbar /= vbar.sum();
    vbar.swap(v);
    niter++;
  } while(niter<MAXITER);
  return v;
}

MatrixXd Gmatrix::compute_Grr()
{
  MatrixXd Grr;
  Grr.resize(m_Nr, m_Nr);

  for(int i=0; i<m_Nr; i++)
  {
    for(int j=0; j<m_Nr; j++)
    {
      Grr(i,j) = get_G(m_rnodes[i],m_rnodes[j]);
    }
  }
  return Grr;
}

void Gmatrix::Project_rnodes(VectorXd& Psi)
{
  for(int i=0; i<m_Nr; i++)
  {
    Psi[m_rnodes[i]]=0;
  }
}

double Gmatrix::compute_PsiR(int MAXITER)
{
  double lambda_c = 0.0;
  VectorXd PsiR = VectorXd::Zero(m_N);
  VectorXd v = VectorXd::Constant(m_N, 1.0/(double)m_N);
  Project_rnodes(v);
  int niter=0;
  do
  {
    power_iteration(v, PsiR);
    PsiR /= PsiR.sum();
    Project_rnodes(PsiR);
    lambda_c = PsiR.sum();
    PsiR /= lambda_c;
    niter++;
    PsiR.swap(v);
  } while(niter<MAXITER);
  m_lambda_c = lambda_c;
  m_PsiR = PsiR;
  return lambda_c;
}

double Gmatrix::compute_PsiL(int MAXITER)
{
  double dlambda;
  VectorXd PsiL = VectorXd::Zero(m_N);
  VectorXd v = VectorXd::Constant(m_N, 1.0);
  int i, j, niter = 0;
  Project_rnodes(v);

  do
  {
    double dangling = 0.0;
    double sumvi = 0.0;
    double inv = 1.0/(double)m_N;
    for(i=0; i<m_N; i++)
    {
      sumvi += v[i];
      PsiL[i] = 0.0;
    }
    dangling=inv*m_alpha*sumvi;
    for(i=0; i<m_N; i++)
    {
      if(m_Smatrix[i].nbOut==0)
      {
        PsiL[i] = dangling;
      }
      else
      {
        for(j=0; j<m_Smatrix[i].nbOut; j++)
        {
          PsiL[i]+=m_alpha*v[m_Smatrix[i].outLinks.first[j]]*m_Smatrix[i].outLinks.second[j]/m_Smatrix[i].sumWeight;
        }
      }
    }
    PsiL = PsiL.array() + ((1.0-m_alpha)/(double)m_N)*sumvi;

    Project_rnodes(PsiL);
    dlambda = m_PsiR.dot(PsiL);
    PsiL /= dlambda;
    niter++;
    v = PsiL;

  } while(niter<MAXITER);

  PsiL /= m_PsiR.dot(PsiL) ;
  m_PsiL = PsiL;
  return dlambda;
}

MatrixXd Gmatrix::compute_Gpr()
{
  MatrixXd Gpr;
  Gpr.resize(m_Nr, m_Nr);
  int i, j;
  VectorXd PsiR_tilde = VectorXd::Zero(m_Nr);
  RowVectorXd PsiL_tilde = RowVectorXd::Zero(m_Nr);
  for(i=0; i<m_Nr; i++)
  {
    for(j=0; j<m_N; j++)
    {
      PsiR_tilde[i] += get_G(m_rnodes[i],j)*m_PsiR[j];
      PsiL_tilde[i] += get_G(j,m_rnodes[i])*m_PsiL[j];
    }
  }
  Gpr = (PsiR_tilde*PsiL_tilde).array()/(1.0-m_lambda_c);
  return Gpr;
}

void Gmatrix::Project_Pc(VectorXd& vector)
{
  double sp = vector.dot(m_PsiL);
  vector = sp * m_PsiR;
}

void Gmatrix::Project_Qc(VectorXd& vector)
{
  double sp = vector.dot(m_PsiL);
  vector -= sp*m_PsiR;
}

MatrixXd Gmatrix::compute_Gqr(int MAXITER)
{
  MatrixXd Gqr;
  Gqr.resize(m_Nr, m_Nr);
  int i,j,k;

  VectorXd col = VectorXd::Zero(m_N);
  for(i=0; i<m_Nr; i++) //For the ith column of Gqr
  {
    //get the column in Gsr
    for(j=0; j<m_N; j++)
    {
      col[j] = get_G(j,m_rnodes[i]);
    }
    Project_rnodes(col);
    Project_Qc(col);

    VectorXd col_temp = VectorXd::Zero(m_N);
    VectorXd col_res = VectorXd::Zero(m_N);
    //col_res is the result of Q_c*\sum-{l=0}^{\infty}*Gsi, i being the ith Gsr column considered
    col_res = col;
    int niter=0;
    do
    {
      power_iteration(col, col_temp);
      Project_rnodes(col_temp);
      Project_Qc(col_temp);
      niter++;
      col_res += col_temp;

      col_temp.swap(col);
    } while(niter<MAXITER);

    // Now the product with Grs
    VectorXd Gqr_col = VectorXd::Zero(m_Nr);
    for(j=0; j<m_Nr; j++)
    {
      for(k=0; k<m_N; k++)
      {
        Gqr_col[j] += get_G(m_rnodes[j], k)*col_res[k];
      }
    }
    
    Gqr(Eigen::all,i) = Gqr_col;
  }
  return Gqr;
}


vector<MatrixXd> Gmatrix::compute_Regomax(vector<int> rnodes, int MAXITER)
{
  vector<MatrixXd> res;
  m_rnodes = rnodes;
  m_Nr = m_rnodes.size();

  res.push_back(Gmatrix::compute_Grr());
  Gmatrix::compute_PsiR(MAXITER);
  Gmatrix::compute_PsiL(MAXITER);
  res.push_back(Gmatrix::compute_Gpr());
  res.push_back(Gmatrix::compute_Gqr(MAXITER));

  return res;
}

PYBIND11_MODULE(pagerankcpp, m) 
{
  m.doc() = "PageRank and Regomax with cpp";

  py::class_<Gmatrix>(m, "Gmatrix")
      .def(py::init<vector<int>, vector<int>, vector<double>, double>(),
              py::arg("nfrom"), py::arg("to"), py::arg("weight"), py::arg("alpha"))
      .def("pagerank", &Gmatrix::compute_PageRank, py::arg("MAXITER"))
      .def("regomax", &Gmatrix::compute_Regomax, py::arg("rnodes"), py::arg("MAXITER"));
}