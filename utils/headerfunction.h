

  TH1D* RescaleByMean(TH1D* h1){
    Double_t val,mean,error;
    Int_t nBinX = h1->GetNbinsX();
    const char *name1 = h1->GetName();
    TH1D *hnew  = (TH1D *) h1->Clone(Form("%s_new",name1));
    hnew->Reset();
    
    for(int i=1; i<=nBinX; i++){
      mean = h1->GetBinCenter(i);
      val  =  h1->GetBinContent(i);
      error =  h1->GetBinError(i);
      if(val==0 && error==0) continue;
      hnew->SetBinContent(i,val/mean);
      hnew->SetBinError(i,error/mean);
    }
    return hnew;
  }






TCanvas *GetCanvas(TString title,int xpos,int ypos,int sizeX,int sizeY,Bool_t gridx,Bool_t gridy,float topMgn,float botMgn,float leftMgn,float rightMgn)
{
  TCanvas *c1 = new TCanvas(title,title,xpos,ypos,sizeX,sizeY);
  //c1->SetCanvasSize(sizeX,sizeY);
  //c1->SetTitle(title);
  c1->SetTopMargin(topMgn);
  c1->SetRightMargin(rightMgn);
  c1->SetLeftMargin(leftMgn);
  c1->SetBottomMargin(botMgn);
  if(gridx)
    c1->SetGridx();
  if(gridy)
    c1->SetGridy();
  return c1;
}

TPad *GetPad(TString name,float xpos1,float ypos1,float xpos2,float ypos2,float topMar,float botMar,float leftMar,float rightMar){
  TPad *tpad = new TPad(name,"",xpos1,ypos1,xpos2,ypos2);
  //tpad->Draw();
  //tpad->cd();
  tpad->SetFillColor(0);
  tpad->SetBorderMode(0);
  tpad->SetBorderSize(2);
  tpad->SetTicks(1,1);
  tpad->SetFrameBorderMode(0);
  tpad->SetFrameBorderMode(0);

  tpad->SetRightMargin(rightMar);
  tpad->SetTopMargin(topMar);
  tpad->SetLeftMargin(leftMar);
  tpad->SetBottomMargin(botMar);
  return tpad;
}



//--------------------------------
void SetLabelSize(TH1 *h1,float yLabelsize,float xLabelsize,float zLabelsize,float offset){
  h1->GetXaxis()->SetLabelFont(42);
  h1->GetXaxis()->SetLabelSize(xLabelsize);
  h1->GetYaxis()->SetLabelFont(42);
  h1->GetYaxis()->SetLabelSize(yLabelsize);
  h1->GetZaxis()->SetLabelFont(42);
  h1->GetZaxis()->SetLabelSize(zLabelsize);
  h1->GetZaxis()->SetLabelOffset(offset);
}

void SetMarkerTH1(TH1 *h1,TString hTitle,int markSyle,float markSize,int markColor,int lineColor){
  h1->SetTitle(hTitle);
  h1->SetMarkerStyle(markSyle);
  h1->SetMarkerSize(markSize);
  h1->SetMarkerColor(markColor);
  h1->SetLineColor(lineColor);
}
void SetAxisTH1(TH1 *h1,float yAxisLow,float yAxisHigh,float xAxisLow,float xAxisHigh,float yLabelsize,float xLabelsize){
  h1->GetYaxis()->SetRangeUser(yAxisLow,yAxisHigh);
  h1->GetXaxis()->SetRangeUser(xAxisLow,xAxisHigh);
  h1->GetXaxis()->SetLabelFont(42);
  h1->GetXaxis()->SetLabelSize(xLabelsize);
  h1->GetYaxis()->SetLabelFont(42);
  h1->GetYaxis()->SetLabelSize(yLabelsize);
}
void SetTitleTH1(TH1 *h1,TString yTitle,float yTileSize,float yOffset,TString xTitle,float xTileSize,float xOffset){
  h1->GetYaxis()->SetTitle(yTitle);
  h1->GetYaxis()->SetTitleSize(yTileSize);
  h1->GetYaxis()->SetTitleOffset(yOffset);
  h1->GetYaxis()->CenterTitle(true);
  h1->GetYaxis()->SetTitleFont(42);
  h1->GetXaxis()->SetTitle(xTitle);
  h1->GetXaxis()->SetTitleSize(xTileSize);
  h1->GetXaxis()->SetTitleOffset(xOffset);
  h1->GetXaxis()->CenterTitle(true);
  h1->GetXaxis()->SetTitleFont(42);
}

//--------------------------------
void SetMarkerTH1(TGraphErrors *h1,TString hTitle,int markSyle,float markSize,int markColor,int lineColor){
  h1->SetTitle(hTitle);
  h1->SetMarkerStyle(markSyle);
  h1->SetMarkerSize(markSize);
  h1->SetMarkerColor(markColor);
  h1->SetLineColor(lineColor);
}
//--------------------------------
void SetMarkerTH1(TH2 *h1,TString hTitle,int markSyle,float markSize,int markColor,int lineColor){
  h1->SetTitle(hTitle);
  h1->SetMarkerStyle(markSyle);
  h1->SetMarkerSize(markSize);
  h1->SetMarkerColor(markColor);
  h1->SetLineColor(lineColor);
}
void SetAxisTH1(TH2 *h1,float yAxisLow,float yAxisHigh,float xAxisLow,float xAxisHigh,float yLabelsize,float xLabelsize){
  h1->GetYaxis()->SetRangeUser(yAxisLow,yAxisHigh);
  h1->GetXaxis()->SetRangeUser(xAxisLow,xAxisHigh);
  h1->GetXaxis()->SetLabelFont(42);
  h1->GetXaxis()->SetLabelSize(xLabelsize);
  h1->GetYaxis()->SetLabelFont(42);
  h1->GetYaxis()->SetLabelSize(yLabelsize);
}

void SetAxisTH1(TH2 *h1,float yAxisLow,float yAxisHigh,float xAxisLow,float xAxisHigh,float yLabelsize,float xLabelsize,float zLabelsize,float zLabOffset){
  h1->GetYaxis()->SetRangeUser(yAxisLow,yAxisHigh);
  h1->GetXaxis()->SetRangeUser(xAxisLow,xAxisHigh);
  h1->GetXaxis()->SetLabelFont(42);
  h1->GetXaxis()->SetLabelSize(xLabelsize);
  h1->GetYaxis()->SetLabelFont(42);
  h1->GetYaxis()->SetLabelSize(yLabelsize);
  h1->GetZaxis()->SetLabelFont(42);
  h1->GetZaxis()->SetLabelSize(zLabelsize);
  h1->GetZaxis()->SetLabelOffset(zLabOffset);  
  }

void SetTitleTH1(TH2 *h1,TString yTitle,float yTileSize,float yOffset,TString xTitle,float xTileSize,float xOffset){
  h1->GetYaxis()->SetTitle(yTitle);
  h1->GetYaxis()->SetTitleSize(yTileSize);
  h1->GetYaxis()->SetTitleOffset(yOffset);
  h1->GetYaxis()->CenterTitle(true);
  h1->GetYaxis()->SetTitleFont(42);
  h1->GetXaxis()->SetTitle(xTitle);
  h1->GetXaxis()->SetTitleSize(xTileSize);
  h1->GetXaxis()->SetTitleOffset(xOffset);
  h1->GetXaxis()->CenterTitle(true);
  h1->GetXaxis()->SetTitleFont(42);
}



void drawBox(Float_t xstart,Float_t ystart,Float_t xstop,Float_t ystop, Int_t color, Int_t style){

  /// bottom edge:
  TLine *line1 = new TLine(xstart,ystart,xstop,ystart);
  line1->SetLineColor(color);
  line1->SetLineStyle(style);
  line1->Draw();
  /// top edge:
  TLine *line2 = new TLine(xstart,ystop,xstop,ystop);
  line2->SetLineColor(color);
  line2->SetLineStyle(style);
  line2->Draw();
  /// left edge:
  TLine *line3 = new TLine(xstart,ystart,xstart,ystop);
  line3->SetLineColor(color);
  line3->SetLineStyle(style);
  line3->Draw();
  /// right edge:
  TLine *line4 = new TLine(xstop,ystart,xstop,ystop);
  line4->SetLineColor(color);
  line4->SetLineStyle(style);
  line4->Draw();  
}


Double_t thetaFromEta(Double_t eta){
  return 2.*TMath::ATan(TMath::Exp(-1.*eta));
}



void drawMyline(Float_t xstrt,Float_t ystrt,Float_t xend,Float_t yend,Int_t iStyle,Int_t iWidth,Int_t icol){

  TLine *line = new TLine(xstrt, ystrt, xend, yend);
  line->SetLineStyle(iStyle);
  line->SetLineWidth(iWidth);
  line->SetLineColor(icol);
  line->Draw();
 
}

void drawMyText(Float_t xPos, Float_t yPos, Float_t size, TString text){
  TLatex *tex = new TLatex(xPos,yPos,text.Data());
  tex->SetTextFont(42);
  tex->SetTextSize(size);
  tex->SetLineWidth(2);
  tex->Draw();
}

void drawMyTextNDC(Float_t xPos, Float_t yPos, Float_t size, TString text){
  TLatex *tex = new TLatex();
  tex->SetTextFont(42);
  tex->SetTextSize(size);
  tex->SetLineWidth(2);
  tex->DrawLatexNDC(xPos,yPos,text.Data());
}
void drawMyTextNDC(Float_t xPos, Float_t yPos, Float_t size, TString text,Color_t col){
  TLatex *tex = new TLatex();
  tex->SetTextFont(42);
  tex->SetTextSize(size);
  tex->SetLineWidth(2);
  tex->SetTextColor(col);
  tex->DrawLatexNDC(xPos,yPos,text.Data());
}







