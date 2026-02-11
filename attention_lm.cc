#include <cstdlib>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <unordered_map>
#include <random>
#include <cmath>

using namespace Eigen;
using namespace std;

// HyperParameters
const int BLOCK_SIZE = 32;
const int BATCH_SIZE = 8;
const int N_EMBD = 32;
const float LR = 3e-4;

//Adam HyperParameters
const float BETA1 = 0.9f;
const float BETA2 = 0.999f;
const float EPS = 1e-8f;

//FFN Parameters
const int FFN_HIDDEN = 4 * N_EMBD;
MatrixXf W_ff1;
RowVectorXf b_ff1;
MatrixXf W_ff2;
RowVectorXf b_ff2;

struct FFNStats{
  float mean = 0.0f;
  float max = 0.0f;
  float min = 0.0f;
};

// Utilities
float randf(){
  static random_device rd;
  static mt19937 gen(rd());
  static normal_distribution<float> dist(0.0f, 0.02f / sqrt(N_EMBD));
  return dist(gen);
}

VectorXf softmax(const VectorXf& x){
  VectorXf y = x.array() - x.maxCoeff();
  y = y.array().exp();
  float s = max(y.sum(), 1e-9f);
  if(!isfinite(s) || s < 1e-9f)
    s = 1e-9f;
  return y/s;
}

inline float gelu(float x){
  return 0.5 * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

inline float gelu_grad(float x){
  float u = sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x);
  float tanh_u = tanh(u);
  float du_dx = sqrt(2.0f / M_PI) * (1 + 3 * 0.044715f * x * x);
  return 0.5f * (1 + tanh_u) + 0.5f * x * (1 - tanh_u * tanh_u) * du_dx;
}

MatrixXf gelu(const MatrixXf& X){
  MatrixXf Y = X;
  for(int i = 0; i < X.size(); i++)
    Y.data()[i] = gelu(X.data()[i]);
  return Y;
}

//Vocabulary
vector<char> vocab;
unordered_map<char, int> STOI;
unordered_map<int, char> ITOS;

//Model Parameters
// Embedding Table
MatrixXf W_embed;
//Attention weights
MatrixXf W_q, W_k, W_v;
//Output projection
MatrixXf W_out;
//Positional Embedding
MatrixXf W_pos;

//Layer Norm
MatrixXf LN_gamma, LN_beta;

//Adam State
MatrixXf m_W_out, v_W_out;
MatrixXf m_W_q, v_W_q;
MatrixXf m_W_k, v_W_k;
MatrixXf m_W_v, v_W_v;
MatrixXf m_W_embed, v_W_embed;
MatrixXf m_W_pos, v_W_pos;
MatrixXf m_LN_gamma, v_LN_gamma;
MatrixXf m_LN_beta, v_LN_beta;
MatrixXf m_W_ff1, v_W_ff1;
RowVectorXf m_b_ff1, v_b_ff1;
MatrixXf m_W_ff2, v_W_ff2;
RowVectorXf m_b_ff2, v_b_ff2;
int adam_t = 0;

//Gradient Buffers
MatrixXf g_W_out, g_W_q, g_W_k, g_W_v;
MatrixXf g_W_embed, g_W_pos;
MatrixXf g_LN_gamma, g_LN_beta;
RowVectorXf g_b_ff1;
MatrixXf g_W_ff1;
RowVectorXf g_b_ff2;
MatrixXf g_W_ff2;

//Load Data
string load_text(const string& path){
  ifstream f(path);
  return string((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
}

void adam_update(MatrixXf& W, MatrixXf& m, MatrixXf& v, const MatrixXf& g,int t){
  m = BETA1 * m + (1 - BETA1) * g;
  v = BETA2 * v + (1 - BETA2) * g.array().square().matrix();

  MatrixXf m_hat = m / (1 - pow(BETA1, t));
  MatrixXf v_hat = v / (1 - pow(BETA2, t));

  W -= LR * (m_hat.array() / (v_hat.array().sqrt() + EPS)).matrix();
}

void adam_update(RowVectorXf& W, RowVectorXf& m, RowVectorXf& v, const RowVectorXf& g, int t){
  m = BETA1 * m + (1 - BETA1) * g;
  v = BETA2 * v + (1 - BETA2) * g.array().square().matrix();

  RowVectorXf m_hat = m / (1 - pow(BETA1, t));
  RowVectorXf v_hat = v / (1 - pow(BETA2, t));

  W -= LR * (m_hat.array() / (v_hat.array().sqrt() + EPS)).matrix();
}

void zero_grads(){
  g_W_embed.setZero();
  g_W_q.setZero();
  g_W_k.setZero();
  g_W_v.setZero();
  g_W_out.setZero();
  g_W_pos.setZero();
  g_LN_gamma.setZero();
  g_LN_beta.setZero();
  g_W_ff1.setZero();
  g_b_ff1.setZero();
  g_W_ff2.setZero();
  g_b_ff2.setZero();
}

void clip(MatrixXf& g){
  g = g.array().max(-1.0f).min(1.0f);
};

void clip(RowVectorXf& g){
  g = g.array().max(-1.0f).min(1.0f);
}

void apply_adam(){
  adam_t++;

  float scale = 1.0f / BATCH_SIZE;

  adam_update(W_embed, m_W_embed, v_W_embed, g_W_embed * scale, adam_t);
  adam_update(W_q, m_W_q, v_W_q, g_W_q * scale, adam_t);
  adam_update(W_k, m_W_k, v_W_k, g_W_k * scale, adam_t);
  adam_update(W_v, m_W_v, v_W_v, g_W_v * scale, adam_t);
  adam_update(W_out, m_W_out, v_W_out, g_W_out * scale, adam_t);
  adam_update(W_pos, m_W_pos, v_W_pos, g_W_pos * scale, adam_t);
  adam_update(LN_gamma, m_LN_gamma, v_LN_gamma, g_LN_gamma * scale, adam_t);
  adam_update(LN_beta, m_LN_beta, v_LN_beta, g_LN_beta * scale, adam_t);
  adam_update(W_ff1, m_W_ff1, v_W_ff1, g_W_ff1 * scale, adam_t);
  adam_update(b_ff1, m_b_ff1, v_b_ff1, g_b_ff1 * scale, adam_t);
  adam_update(W_ff2, m_W_ff2, v_W_ff2, g_W_ff2 * scale, adam_t);
  adam_update(b_ff2, m_b_ff2, v_b_ff2, g_b_ff2 * scale, adam_t);
}

int sample_from_probs(const VectorXf& probs){
  static random_device rd;
  static mt19937 gen(rd());
  discrete_distribution<int> dist(probs.data(), probs.data()+probs.size());
  return dist(gen);
}

void init_model(int vocab_size){
  W_embed = MatrixXf(vocab_size, N_EMBD);
  W_q = MatrixXf(N_EMBD, N_EMBD);
  W_k = MatrixXf(N_EMBD, N_EMBD);
  W_v = MatrixXf(N_EMBD, N_EMBD);
  W_out = MatrixXf(N_EMBD, vocab_size);
  W_pos = MatrixXf(BLOCK_SIZE, N_EMBD);
  LN_gamma = MatrixXf::Ones(1, N_EMBD);
  LN_beta = MatrixXf::Zero(1, N_EMBD);
  W_ff1 = MatrixXf(N_EMBD, FFN_HIDDEN);
  W_ff2 = MatrixXf(FFN_HIDDEN, N_EMBD);
  b_ff1 = RowVectorXf::Zero(FFN_HIDDEN);
  b_ff2 = RowVectorXf::Zero(N_EMBD);

  g_W_embed = MatrixXf::Zero(vocab_size, N_EMBD);
  g_W_q = MatrixXf::Zero(N_EMBD, N_EMBD);
  g_W_k = MatrixXf::Zero(N_EMBD, N_EMBD);
  g_W_v = MatrixXf::Zero(N_EMBD, N_EMBD);
  g_W_out = MatrixXf::Zero(N_EMBD, vocab_size);
  g_W_pos = MatrixXf::Zero(BLOCK_SIZE, N_EMBD);
  g_LN_gamma = MatrixXf::Zero(1, N_EMBD);
  g_LN_beta = MatrixXf::Zero(1, N_EMBD);
  g_W_ff1 = MatrixXf::Zero(N_EMBD, FFN_HIDDEN);
  g_W_ff2 = MatrixXf::Zero(FFN_HIDDEN, N_EMBD);
  g_b_ff1 = RowVectorXf::Zero(FFN_HIDDEN);
  g_b_ff2 = RowVectorXf::Zero(N_EMBD);

  for(int i = 0; i < W_ff1.size(); i++)
    W_ff1.data()[i] = randf();

  for(int i = 0; i < W_ff2.size(); i++)
    W_ff2.data()[i] = randf();

  m_W_out = MatrixXf::Zero(N_EMBD, vocab_size);
  v_W_out = MatrixXf::Zero(N_EMBD, vocab_size);

  m_W_q = MatrixXf::Zero(N_EMBD, N_EMBD);
  v_W_q = MatrixXf::Zero(N_EMBD, N_EMBD);

  m_W_k = MatrixXf::Zero(N_EMBD, N_EMBD);
  v_W_k = MatrixXf::Zero(N_EMBD, N_EMBD);

  m_W_v = MatrixXf::Zero(N_EMBD, N_EMBD);
  v_W_v = MatrixXf::Zero(N_EMBD, N_EMBD);

  m_W_embed = MatrixXf::Zero(vocab_size, N_EMBD);
  v_W_embed = MatrixXf::Zero(vocab_size, N_EMBD);

  m_W_pos = MatrixXf::Zero(BLOCK_SIZE, N_EMBD);
  v_W_pos = MatrixXf::Zero(BLOCK_SIZE, N_EMBD);

  m_LN_gamma = MatrixXf::Zero(1, N_EMBD);
  v_LN_gamma = MatrixXf::Zero(1, N_EMBD);

  m_LN_beta = MatrixXf::Zero(1, N_EMBD);
  v_LN_beta = MatrixXf::Zero(1, N_EMBD);

  m_W_ff1 = MatrixXf::Zero(N_EMBD, FFN_HIDDEN);
  v_W_ff1 = MatrixXf::Zero(N_EMBD, FFN_HIDDEN);

  m_W_ff2 = MatrixXf::Zero(FFN_HIDDEN, N_EMBD);
  v_W_ff2 = MatrixXf::Zero(FFN_HIDDEN, N_EMBD);

  m_b_ff1 = RowVectorXf::Zero(FFN_HIDDEN);
  v_b_ff1 = RowVectorXf::Zero(FFN_HIDDEN);

  m_b_ff2 = RowVectorXf::Zero(N_EMBD);
  v_b_ff2 = RowVectorXf::Zero(N_EMBD);

  for(int i = 0; i < W_embed.size(); i++)
    W_embed.data()[i] = randf();
  for(int i = 0; i < W_q.size(); i++)
    W_q.data()[i] = randf();
  for(int i = 0; i < W_k.size(); i++)
    W_k.data()[i] = randf();
  for(int i = 0; i < W_v.size(); i++)
    W_v.data()[i] = randf();
  for(int i = 0; i < W_out.size(); i++)
    W_out.data()[i] = randf();
  for(int i = 0; i < W_pos.size(); i++)
    W_pos.data()[i] = randf();
}

void layernorm_forward(const MatrixXf& X, MatrixXf& Y, VectorXf& mean, VectorXf& var){
  int T = X.rows();
  int D = X.cols();

  Y = MatrixXf(T, D);
  mean = VectorXf(T);
  var = VectorXf(T);

  for(int t = 0; t < T; t++){
    float m = X.row(t).mean();
    mean(t) = m;

    float v = (X.row(t).array() - m).square().mean();
    var(t) = v;

    VectorXf x_hat = (X.row(t).array() - m) / sqrt(v + EPS);
    Y.row(t) = x_hat.transpose().array() * LN_gamma.row(0).array() + LN_beta.row(0).array();
  }
}

void layernorm_backward(const MatrixXf& X, const MatrixXf& dY, const VectorXf& mean, const VectorXf& var, MatrixXf& dX){
  int T = X.rows();
  int D = X.cols();

  dX = MatrixXf::Zero(T, D);

  for(int t = 0; t < T; t++){
    VectorXf x = X.row(t);
    float m = mean(t);
    float v = var(t);
    float inv_std = 1.0f / sqrt(v + EPS);

    VectorXf x_hat = (x.array() - m) * inv_std;
    VectorXf dy = dY.row(t);

    //gamma & beta grads
    g_LN_gamma.row(0) += (dy.array() * x_hat.array()).matrix().transpose();
    g_LN_beta.row(0) += dy.transpose();

    float sum_dy = dy.sum();
    float sum_dy_xhat = (dy.array() * x_hat.array()).sum();

    VectorXf gamma = LN_gamma.row(0).transpose();

    VectorXf dx = (gamma.array() * inv_std / D) * (D * dy.array() - sum_dy - x_hat.array() * sum_dy_xhat);

    dX.row(t) += dx.transpose();
  }
}

float train_step(const vector<int>& context, int target, FFNStats* stats = nullptr){
  int T = context.size();

  //Forward
  MatrixXf X(T, N_EMBD);
  for(int i = 0; i < T; i++)
    X.row(i) = W_embed.row(context[i]) + W_pos.row(i);

  //LN1
  MatrixXf Xn;
  VectorXf ln1_mean, ln1_var;
  layernorm_forward(X, Xn, ln1_mean, ln1_var);

  MatrixXf Q = Xn * W_q;
  MatrixXf K = Xn * W_k;
  MatrixXf V = Xn * W_v;

  MatrixXf scores = (Q * K.transpose()) / sqrt(N_EMBD);

  for(int i = 0; i < T; i++)
    for(int j = i + 1; j < T; j++)
      scores(i, j) = -1e9;

  MatrixXf attn(T, T);
  for(int i = 0; i < T; i++)
    attn.row(i) = softmax(scores.row(i).transpose());

  MatrixXf A = attn * V;

  //Residual + LN2
  MatrixXf A_res = A + X;

  MatrixXf A_norm;
  VectorXf ln2_mean, ln2_var;
  layernorm_forward(A_res, A_norm, ln2_mean, ln2_var);

  //FFN Forward
  MatrixXf ff1 = A_norm * W_ff1;
  ff1.rowwise() += b_ff1.row(0);

  MatrixXf ff1_act = gelu(ff1);

  MatrixXf ff2 = ff1_act * W_ff2;
  ff2.rowwise() += b_ff2.row(0);

  MatrixXf FFN_res = ff2 + A_norm;

  MatrixXf FFN_out;
  VectorXf ln3_mean, ln3_var;
  layernorm_forward(FFN_res, FFN_out, ln3_mean, ln3_var);

  if(stats){
    stats -> mean = FFN_out.mean();
    stats -> max = FFN_out.maxCoeff();
    stats -> min = FFN_out.minCoeff();
  }

  //Output head
  VectorXf logits = FFN_out.row(T - 1) * W_out;
  VectorXf probs = softmax(logits);
  float loss = -log(max(probs(target), 1e-9f));

  //Backward

  //Output head
  VectorXf dlogits = probs;
  dlogits(target) -= 1.0f;

  MatrixXf dW_out = FFN_out.row(T - 1).transpose() * dlogits.transpose();
  VectorXf dFFN_out_last = W_out * dlogits;

  MatrixXf dFFN_out = MatrixXf::Zero(T, N_EMBD);
  dFFN_out.row(T - 1) = dFFN_out_last.transpose();

  //LN3 backward
  MatrixXf dFFN_res;
  layernorm_backward(FFN_res, dFFN_out, ln3_mean, ln3_var, dFFN_res);

  //Residual Split
  MatrixXf dff2 = dFFN_res;
  MatrixXf dA_fnn = dFFN_res;

  //FFN linear 2
  MatrixXf dW_ff2 = ff1_act.transpose() * dff2;
  RowVectorXf db_ff2 = dff2.colwise().sum();
  MatrixXf dff1_act = dff2 * W_ff2.transpose();

  //GELU Backward
  MatrixXf dff1 = dff1_act;
  for(int i = 0; i < dff1.size(); i++)
    dff1.data()[i] *= gelu_grad(ff1.data()[i]);

  // FFN linear 1
  MatrixXf dW_ff1 = A_norm.transpose() * dff1;
  RowVectorXf db_ff1 = dff1.colwise().sum();
  MatrixXf dA_from_ffn = dff1 * W_ff1.transpose();

  //Accumulate FFN grads
  g_W_ff1 += dW_ff1;
  g_b_ff1 += db_ff1;
  g_W_ff2 += dW_ff2;
  g_b_ff2 += db_ff2;

  MatrixXf dA_norm = dA_fnn + dA_from_ffn;
  
  //LN2 Backward
  MatrixXf dA_res;
  layernorm_backward(A_res, dA_norm, ln2_mean, ln2_var, dA_res);

  //Residual Split
  MatrixXf dA = dA_res;
  MatrixXf dX = dA_res;

  //Attention Backward
  MatrixXf dAttn = dA * V.transpose();
  MatrixXf dV = attn.transpose() * dA;

  MatrixXf dScores = MatrixXf::Zero(T, T);
  for(int i = 0; i < T; i++){
    VectorXf s = attn.row(i).transpose();
    VectorXf ds = dAttn.row(i).transpose();
    float dot = ds.dot(s);
    dScores.row(i) = (s.array() * (ds.array() - dot)).transpose();
  }

  for(int i = 0; i < T; i++)
    for(int j = i + 1; j < T; j++)
      dScores(i, j) = 0.0f;

  dScores /= sqrt(N_EMBD);

  MatrixXf dQ = dScores * K;
  MatrixXf dK = dScores.transpose() * Q;

  MatrixXf dW_q = Xn.transpose() * dQ;
  MatrixXf dW_k = Xn.transpose() * dK;
  MatrixXf dW_v = Xn.transpose() * dV;
 
  dX += 
    dQ * W_q.transpose() + 
    dK * W_k.transpose() + 
    dV * W_v.transpose();

  //LN1 backward
  MatrixXf dX_ln;
  layernorm_backward(X, dX, ln1_mean, ln1_var, dX_ln);
  dX = dX_ln;

  //Embeddings
  MatrixXf dW_embed = MatrixXf::Zero(W_embed.rows(), W_embed.cols());
  for(int i = 0; i < T; i++)
    dW_embed.row(context[i]) += dX.row(i);

  MatrixXf dW_pos = MatrixXf::Zero(W_pos.rows(), W_pos.cols());
  for(int i = 0; i < T; i++)
    dW_pos.row(i) += dX.row(i);

  //Accumulate
  g_W_embed += dW_embed;
  g_W_q += dW_q;
  g_W_k += dW_k;
  g_W_v += dW_v;
  g_W_out += dW_out;
  g_W_pos += dW_pos;
 
  return loss;
}

void generate(int start_token, int steps){
  vector<int> ctx = {start_token};

  for(int i = 0; i < steps; i++){
    int T = min((int)ctx.size(), BLOCK_SIZE);
    vector<int> window(ctx.end() - T, ctx.end());

    MatrixXf X(T, N_EMBD);
    for(int j = 0; j < T; j++)
      X.row(j) = W_embed.row(window[j]) + W_pos.row(j);

    MatrixXf Xn;
    VectorXf mean, var;
    layernorm_forward(X, Xn, mean, var);

    MatrixXf Q = Xn * W_q;
    MatrixXf K = Xn * W_k;
    MatrixXf V = Xn * W_v;

    MatrixXf scores = (Q * K.transpose()) / sqrt(N_EMBD);
    for(int a = 0; a < T; a++)
      for(int b = a + 1; b < T; b++)
        scores(a, b) = -1e9;

    MatrixXf attn(T, T);
    for(int a  = 0; a < T; a++)
      attn.row(a) = softmax(scores.row(a).transpose());

    MatrixXf A = attn * V;
    VectorXf logits = A.row(T - 1) * W_out;
    logits /= 0.8f; //temperature 
    VectorXf probs = softmax(logits);

    int next = sample_from_probs(probs);

    cout << ITOS[next];
    ctx.push_back(next);
  }
  cout << endl;
}

int main (int argc, char *argv[]) {
  string text = load_text("input.txt");

  for(char c : text){
    if(!STOI.count(c)){
      int id = vocab.size();
      vocab.push_back(c);
      STOI[c] = id;
      ITOS[id] = c;
    }
  }

  init_model(vocab.size());

  //Training Loop
  float avg_loss = 0.0f;
  FFNStats stats;

  for(int step = 0; step < 30000; step++){
    zero_grads();
    float batch_loss = 0.0f;

    for(int b = 0; b < BATCH_SIZE; b++){
      int idx = rand() % (text.size() - BLOCK_SIZE - 1);

      vector<int> ctx;
      for(int j = 0; j < BLOCK_SIZE; j++)
        ctx.push_back(STOI[text[idx + j]]);

      int target = STOI[text[idx + BLOCK_SIZE]];

      batch_loss += train_step(ctx, target, (step % 1000 == 0 && b == 0) ? &stats : nullptr);
    }

    clip(g_W_q);
    clip(g_W_k);
    clip(g_W_v);
    clip(g_W_out);
    clip(g_W_embed);
    clip(g_W_pos);
    clip(g_W_ff1);
    clip(g_b_ff1);
    clip(g_W_ff2);
    clip(g_b_ff2);

    apply_adam();
    avg_loss += batch_loss / BLOCK_SIZE;

    if(step % 1000 == 0 && step > 0){
      cout << "Step " << step << " | avg loss: " << avg_loss / 1000 << " | FFN mean: " << stats.mean << " | FFN max: " << stats.max << " | FFN min: " << stats.min << endl;
      avg_loss = 0.0f;
    }
  }

  cout << "\n--- Generated ---\n";
  generate(STOI[text[0]], 300);

  return 0;
}

