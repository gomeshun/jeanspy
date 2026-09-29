// Experiment-local IEEE binary128 reference of the frozen quadrature target.
// Six forward derivatives; no rank cut, pseudo-determinant or jitter.
#include <quadmath.h>
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#ifdef EXTENDED
using Q=long double;
#define expq expl
#define logq logl
#define log1pq log1pl
#define sqrtq sqrtl
#define fabsq fabsl
#define acosq acosl
#define acoshq acoshl
#define coshq coshl
#define powq powl
#define erfq erfl
#define finiteq std::isfinite
#undef HUGE_VALQ
#define HUGE_VALQ HUGE_VALL
#else
using Q=__float128;
#endif
struct D {Q v; std::array<Q,6> d{}; D(Q x=0):v(x){} };
D operator+(D a,D b){D r(a.v+b.v);for(int k=0;k<6;k++)r.d[k]=a.d[k]+b.d[k];return r;}
D operator-(D a,D b){D r(a.v-b.v);for(int k=0;k<6;k++)r.d[k]=a.d[k]-b.d[k];return r;}
D operator-(D a){return D(0)-a;}
D operator*(D a,D b){D r(a.v*b.v);for(int k=0;k<6;k++)r.d[k]=a.d[k]*b.v+a.v*b.d[k];return r;}
D operator/(D a,D b){D r(a.v/b.v);for(int k=0;k<6;k++)r.d[k]=(a.d[k]-r.v*b.d[k])/b.v;return r;}
D ex(D a){D r(expq(a.v));for(int k=0;k<6;k++)r.d[k]=r.v*a.d[k];return r;}
D lg(D a){D r(logq(a.v));for(int k=0;k<6;k++)r.d[k]=a.d[k]/a.v;return r;}
D soft(D a){Q sig=1/(1+expq(-a.v));D r(a.v>0?a.v+log1pq(expq(-a.v)):log1pq(expq(a.v)));for(int k=0;k<6;k++)r.d[k]=sig*a.d[k];return r;}
D clip(D a,Q low,Q high){return a.v<low?D(low):(a.v>high?D(high):a);}
Q logvol(std::vector<std::array<Q,6>> a,bool house){
 int n=a.size();Q out=0;
 for(int j=0;j<6;j++){Q s=0;for(auto &x:a)s+=x[j]*x[j];s=sqrtq(s);if(s==0)return -HUGE_VALQ;out+=logq(s);for(auto &x:a)x[j]/=s;}
 if(house){
  for(int j=0;j<6;j++){
   Q norm=0;for(int i=j;i<n;i++)norm+=a[i][j]*a[i][j];norm=sqrtq(norm);
   if(norm==0)return -HUGE_VALQ;out+=logq(norm);
   std::vector<Q>v(n-j);for(int i=j;i<n;i++)v[i-j]=a[i][j];v[0]+=(a[j][j]>=0?norm:-norm);
   Q vv=0;for(Q x:v)vv+=x*x;
   for(int k=j+1;k<6;k++){Q p=0;for(int i=j;i<n;i++)p+=v[i-j]*a[i][k];p*=2/vv;for(int i=j;i<n;i++)a[i][k]-=p*v[i-j];}
  }
 }else{
  for(int j=0;j<6;j++){
   for(int pass=0;pass<2;pass++)for(int k=0;k<j;k++){Q p=0;for(int i=0;i<n;i++)p+=a[i][k]*a[i][j];for(int i=0;i<n;i++)a[i][j]-=p*a[i][k];}
   Q s=0;for(auto &x:a)s+=x[j]*x[j];s=sqrtq(s);if(s==0)return -HUGE_VALQ;out+=logq(s);for(auto &x:a)x[j]/=s;
  }
 }
 return out;
}
extern "C" int evaluate(int nr,int nu,int nm,int nk,double umax,double re,const double *theta,const double *radii,const double *errors,const double *y,const double *mn,const double *mw,const double *kn,const double *kw,double *out,double *variance,double *jac){
 const Q ln10=logq(10),pi=acosq(-1),sq2=sqrtq(2),G=Q(1.32712440018e20)/Q(3.085677581491367e16)*Q(1e-6);
 D t[6];for(int k=0;k<6;k++){t[k]=D(Q(theta[k]));t[k].d[k]=1;}
 D rho=ex(ln10*t[0]),rs=ex(ln10*t[1]),a=t[2],b=t[3],g=t[4],beta=1-ex(-ln10*t[5]),p=3-g,q=(b-g)/a;
 Q xmax=sqrtq(logq(umax)),h=xmax/(nu-1);
 std::vector<Q>u(nu),sw(nu,0);std::vector<D>kernel(nu);
 int last=nu%2?nu-1:nu-2;
 for(int j=0;j<=last;j++)sw[j]=h/3*((j==0||j==last)?1:(j%2?4:2));
 if(nu%2==0){sw[nu-2]+=h/2;sw[nu-1]+=h/2;}
 for(int j=0;j<nu;j++){
  Q xx=j*h;u[j]=expq(xx*xx);if(j==0)continue;
  Q smax=acoshq(u[j]);D val;
  for(int k=0;k<nk;k++){Q ch=coshq(smax*Q(kn[k]));D lr=clip(2*beta*(logq(u[j])-logq(ch)),-80,80);val=val+Q(kw[k])*ch*(1-beta/(ch*ch))*ex(lr);}
  kernel[j]=val*(smax/u[j]);
 }
 std::vector<std::array<Q,6>> mat(nr);std::vector<Q>sv(nr);Q info=0,ysum=0;
 for(int i=0;i<nr;i++){
  D total;Q R=radii[i],Re=re;
  for(int j=1;j<nu;j++){
   Q r=R*u[j];D x=D(r)/rs,lc=x.v<1?lg(x):D(0),len=x.v>1?lg(x):D(0),inner,outer;
   for(int k=0;k<nm;k++){
    Q node=mn[k],w=mw[k];
    D shape=ex(-q*soft(a*(lc+4/p*logq(node))));inner=inner+w*node*node*node*shape;
    D z=len*node;outer=outer+w*ex(p*z-q*soft(a*z));
   }
   D mass=4*pi*rho*rs*rs*rs*(4/p*ex(p*lc)*inner+len*outer);
   Q ratio=3/(4*Re)*powq(1+(r/Re)*(r/Re),-Q(2.5))*powq(1+(R/Re)*(R/Re),2);
   total=total+sw[j]*2*kernel[j]*ratio*G*mass*(2*j*h);
  }
  D S=clip(total,Q(1e-12),Q(1e12))+Q(errors[i])*Q(errors[i]);sv[i]=S.v;info+=1/S.v;ysum+=Q(y[i])/S.v;variance[i]=(double)total.v;
  for(int k=0;k<6;k++){mat[i][k]=S.d[k]/S.v/sq2;jac[i*6+k]=(double)(S.d[k]/S.v);}
 }
 Q mu=ysum/info,peak=0;
 for(int i=0;i<nr;i++)peak-=Q(.5)*(logq(2*pi*sv[i])+(Q(y[i])-mu)*(Q(y[i])-mu)/sv[i]);
 Q z=(erfq((1000-mu)*sqrtq(info/2))+erfq((mu+1000)*sqrtq(info/2)))/2;
 Q constant=-logq(Q(8)*5*Q(2.5)*7*Q(1.2)*2*2000);
 Q marginal=peak+logq(2*pi/info)/2+logq(z)+constant;
 Q lv=logvol(mat,false),lv2=logvol(mat,true),lp=lv+logq(info)/2;
 out[0]=(double)(marginal+lp);out[1]=(double)lp;out[2]=(double)mu;out[3]=(double)(1/sqrtq(info));out[4]=(double)fabsq(lv-lv2);out[5]=(double)marginal;
 return finiteq(out[0])?0:1;
}
