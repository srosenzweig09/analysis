/*
 * Author: Suzanne Rosenzweig, s.rosenzweig@cern.ch
 * Purpose: To filter the selected events
 */

#include <iostream>
#include "TH1F.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TChain.h"
#include <cstdlib>
#include <vector>
#include "TString.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

int prepare_training_file(){

  TFile *fout = new TFile("inputs/NMSSM_XYH_YToHH_6b_MX_700_MY_400_training_set_small_batch.root","RECREATE");
  TTree *t1   = new TTree("sixBtree","sixBtree");

  TString tree = "sixBtree";
  TChain *cc  = new TChain(tree);

  int numFiles = 90;
  for (int fileNum=0;fileNum < numFiles;fileNum++) {
    // cc->AddFile(Form("root://cmseos.fnal.gov//store/user/srosenzw/sixb_ntuples/preselections/NMSSM_XYH_YToHH_6b_MX_700_MY_400/output/ntuple_%d.root", fileNum));
    // cc->AddFile(Form("root://cmseos.fnal.gov//store/user/ekoenig/6BAnalysis/NTuples/2018/SR/NMSSM/NMSSM_XYH_YToHH_6b_MX_700_MY_400_10M/training/ntuple_%d.root", fileNum));
    cc->AddFile(Form("root://cmseos.fnal.gov//store/user/ekoenig/6BAnalysis/NTuples/2018/SR/NN/NMSSM/NMSSM_XYH_YToHH_6b_MX_700_MY_400_10M/training/ntuple_%d.root", fileNum));
  }

  TTreeReader reader(cc);

  TTreeReaderValue<int> n_sixb(reader,"nfound_sixb");
  TTreeReaderValue<int> n_jet(reader,"n_jet");

  TTreeReaderArray<float> jet_pt(reader,"jet_pt");
  TTreeReaderArray<float> jet_eta(reader,"jet_eta");
  TTreeReaderArray<float> jet_phi(reader,"jet_phi");
  TTreeReaderArray<float> jet_mass(reader,"jet_m");
  TTreeReaderArray<float> jet_btag(reader,"jet_btag");
  TTreeReaderArray<float> jet_qgl(reader,"jet_qgl");
  TTreeReaderArray<int> jet_idx(reader,"jet_signalId");
  // TTreeReaderArray<int> jet_hadronFlav(reader,"jet_hadronFlav");
  // TTreeReaderArray<int> jet_partonFlav(reader,"jet_partonFlav");
        
  TTreeReaderValue<float> gen_HX_b1_recojet_m(reader,"gen_HX_b1_recojet_m");
  TTreeReaderValue<float> gen_HX_b1_recojet_pt(reader,"gen_HX_b1_recojet_pt");
  TTreeReaderValue<float> gen_HX_b1_recojet_eta(reader,"gen_HX_b1_recojet_eta");
  TTreeReaderValue<float> gen_HX_b1_recojet_phi(reader,"gen_HX_b1_recojet_phi");
  TTreeReaderValue<float> gen_HX_b1_recojet_ptRegressed(reader,"gen_HX_b1_recojet_ptRegressed");

  TTreeReaderValue<float> gen_HX_b2_recojet_m(reader,"gen_HX_b2_recojet_m");
  TTreeReaderValue<float> gen_HX_b2_recojet_pt(reader,"gen_HX_b2_recojet_pt");
  TTreeReaderValue<float> gen_HX_b2_recojet_eta(reader,"gen_HX_b2_recojet_eta");
  TTreeReaderValue<float> gen_HX_b2_recojet_phi(reader,"gen_HX_b2_recojet_phi");
  TTreeReaderValue<float> gen_HX_b2_recojet_ptRegressed(reader,"gen_HX_b2_recojet_ptRegressed");

  TTreeReaderValue<float> gen_HY1_b1_recojet_m(reader,"gen_HY1_b1_recojet_m");
  TTreeReaderValue<float> gen_HY1_b1_recojet_pt(reader,"gen_HY1_b1_recojet_pt");
  TTreeReaderValue<float> gen_HY1_b1_recojet_eta(reader,"gen_HY1_b1_recojet_eta");
  TTreeReaderValue<float> gen_HY1_b1_recojet_phi(reader,"gen_HY1_b1_recojet_phi");
  TTreeReaderValue<float> gen_HY1_b1_recojet_ptRegressed(reader,"gen_HY1_b1_recojet_ptRegressed");

  TTreeReaderValue<float> gen_HY1_b2_recojet_m(reader,"gen_HY1_b2_recojet_m");
  TTreeReaderValue<float> gen_HY1_b2_recojet_pt(reader,"gen_HY1_b2_recojet_pt");
  TTreeReaderValue<float> gen_HY1_b2_recojet_eta(reader,"gen_HY1_b2_recojet_eta");
  TTreeReaderValue<float> gen_HY1_b2_recojet_phi(reader,"gen_HY1_b2_recojet_phi");
  TTreeReaderValue<float> gen_HY1_b2_recojet_ptRegressed(reader,"gen_HY1_b2_recojet_ptRegressed");

  TTreeReaderValue<float> gen_HY2_b1_recojet_m(reader,"gen_HY2_b1_recojet_m");
  TTreeReaderValue<float> gen_HY2_b1_recojet_pt(reader,"gen_HY2_b1_recojet_pt");
  TTreeReaderValue<float> gen_HY2_b1_recojet_eta(reader,"gen_HY2_b1_recojet_eta");
  TTreeReaderValue<float> gen_HY2_b1_recojet_phi(reader,"gen_HY2_b1_recojet_phi");
  TTreeReaderValue<float> gen_HY2_b1_recojet_ptRegressed(reader,"gen_HY2_b1_recojet_ptRegressed");

  TTreeReaderValue<float> gen_HY2_b2_recojet_m(reader,"gen_HY2_b2_recojet_m");
  TTreeReaderValue<float> gen_HY2_b2_recojet_pt(reader,"gen_HY2_b2_recojet_pt");
  TTreeReaderValue<float> gen_HY2_b2_recojet_eta(reader,"gen_HY2_b2_recojet_eta");
  TTreeReaderValue<float> gen_HY2_b2_recojet_phi(reader,"gen_HY2_b2_recojet_phi");
  TTreeReaderValue<float> gen_HY2_b2_recojet_ptRegressed(reader,"gen_HY2_b2_recojet_ptRegressed");

  float HX_b1_recojet_m, HX_b1_recojet_pt, HX_b1_recojet_ptRegressed, HX_b1_recojet_eta, HX_b1_recojet_phi,  HX_b2_recojet_m, HX_b2_recojet_pt, HX_b2_recojet_ptRegressed, HX_b2_recojet_eta, HX_b2_recojet_phi, HY1_b1_recojet_m, HY1_b1_recojet_pt, HY1_b1_recojet_ptRegressed, HY1_b1_recojet_eta, HY1_b1_recojet_phi,  HY1_b2_recojet_m, HY1_b2_recojet_pt, HY1_b2_recojet_ptRegressed, HY1_b2_recojet_eta, HY1_b2_recojet_phi, HY2_b1_recojet_m, HY2_b1_recojet_pt, HY2_b1_recojet_ptRegressed, HY2_b1_recojet_eta, HY2_b1_recojet_phi,  HY2_b2_recojet_m, HY2_b2_recojet_pt, HY2_b2_recojet_ptRegressed, HY2_b2_recojet_eta, HY2_b2_recojet_phi;

  int n_jets, n_sixbs;
  
  std::vector<float> jets_pt, jets_eta, jets_phi, jets_btag, jets_m;

  // std::vector<int> jets_idx, jets_hadronFlav, jets_partonFlav;
  std::vector<int> jets_idx;

  t1->Branch("n_jet",& n_jets);
  t1->Branch("n_sixb",& n_sixbs);

  t1->Branch("jet_pt",& jets_pt);
  t1->Branch("jet_eta",& jets_eta);
  t1->Branch("jet_phi",& jets_phi);
  t1->Branch("jet_m",& jets_m);
  t1->Branch("jet_btag",& jets_btag);
  // t1->Branch("jet_qgl",& jets_qgl);
  t1->Branch("jet_idx",& jets_idx);
  // t1->Branch("jet_hadronFlav",& jets_hadronFlav);
  // t1->Branch("jet_partonFlav",& jets_partonFlav);

  t1->Branch("gen_HX_b1_recojet_m",& HX_b1_recojet_m);
  t1->Branch("gen_HX_b1_recojet_pt",& HX_b1_recojet_pt);
  t1->Branch("gen_HX_b1_recojet_eta",& HX_b1_recojet_eta);
  t1->Branch("gen_HX_b1_recojet_phi",& HX_b1_recojet_phi);
  t1->Branch("gen_HX_b1_recojet_ptRegressed",& HX_b1_recojet_ptRegressed);

  t1->Branch("gen_HX_b2_recojet_m",& HX_b2_recojet_m);
  t1->Branch("gen_HX_b2_recojet_pt",& HX_b2_recojet_pt);
  t1->Branch("gen_HX_b2_recojet_eta",& HX_b2_recojet_eta);
  t1->Branch("gen_HX_b2_recojet_phi",& HX_b2_recojet_phi);
  t1->Branch("gen_HX_b2_recojet_ptRegressed",& HX_b2_recojet_ptRegressed);

  t1->Branch("gen_HY1_b1_recojet_m",& HY1_b1_recojet_m);
  t1->Branch("gen_HY1_b1_recojet_pt",& HY1_b1_recojet_pt);
  t1->Branch("gen_HY1_b1_recojet_eta",& HY1_b1_recojet_eta);
  t1->Branch("gen_HY1_b1_recojet_phi",& HY1_b1_recojet_phi);
  t1->Branch("gen_HY1_b1_recojet_ptRegressed",& HY1_b1_recojet_ptRegressed);

  t1->Branch("gen_HY1_b2_recojet_m",& HY1_b2_recojet_m);
  t1->Branch("gen_HY1_b2_recojet_pt",& HY1_b2_recojet_pt);
  t1->Branch("gen_HY1_b2_recojet_eta",& HY1_b2_recojet_eta);
  t1->Branch("gen_HY1_b2_recojet_phi",& HY1_b2_recojet_phi);
  t1->Branch("gen_HY1_b2_recojet_ptRegressed",& HY1_b2_recojet_ptRegressed);

  t1->Branch("gen_HY2_b1_recojet_m",& HY2_b1_recojet_m);
  t1->Branch("gen_HY2_b1_recojet_pt",& HY2_b1_recojet_pt);
  t1->Branch("gen_HY2_b1_recojet_eta",& HY2_b1_recojet_eta);
  t1->Branch("gen_HY2_b1_recojet_phi",& HY2_b1_recojet_phi);
  t1->Branch("gen_HY2_b1_recojet_ptRegressed",& HY2_b1_recojet_ptRegressed);

  t1->Branch("gen_HY2_b2_recojet_m",& HY2_b2_recojet_m);
  t1->Branch("gen_HY2_b2_recojet_pt",& HY2_b2_recojet_pt);
  t1->Branch("gen_HY2_b2_recojet_eta",& HY2_b2_recojet_eta);
  t1->Branch("gen_HY2_b2_recojet_phi",& HY2_b2_recojet_phi);
  t1->Branch("gen_HY2_b2_recojet_ptRegressed",& HY2_b2_recojet_ptRegressed);

  int eventCount = 0;
  int passCount = 0;
  while(reader.Next()){
    eventCount++;
    if (eventCount % 10000 == 0) {
      std::cout << eventCount << " events read!" << std::endl;
    }
    if (eventCount > 100000) {continue;} 

    // if (*n_sixb != 6) continue;
    // if (*n_jet < 7) continue;

    jets_idx.clear();
    jets_pt.clear();
    jets_eta.clear();
    jets_phi.clear();
    jets_m.clear();
    jets_btag.clear();
    // jets_qgl.clear();
    // jets_partonFlav.clear();
    // jets_hadronFlav.clear();

    passCount++;

    for (unsigned int i=0; i<jet_pt.GetSize(); i++){
      jets_pt.emplace_back(jet_pt[i]);
      jets_eta.emplace_back(jet_eta[i]);
      jets_phi.emplace_back(jet_phi[i]);
      jets_m.emplace_back(jet_mass[i]);
      jets_btag.emplace_back(jet_btag[i]);
      // jets_qgl.emplace_back(jet_qgl[i]);
      jets_idx.emplace_back(jet_idx[i]);
      // jets_hadronFlav.emplace_back(jet_hadronFlav[i]);
      // jets_partonFlav.emplace_back(jet_partonFlav[i]);
    }

    n_jets = *n_jet;
    n_sixbs = *n_sixb;

    HX_b1_recojet_m   = *gen_HX_b1_recojet_m;
    HX_b1_recojet_pt  = *gen_HX_b1_recojet_pt;
    HX_b1_recojet_eta = *gen_HX_b1_recojet_eta;
    HX_b1_recojet_phi = *gen_HX_b1_recojet_phi;
    HX_b1_recojet_ptRegressed = *gen_HX_b1_recojet_ptRegressed;

    HX_b2_recojet_m   = *gen_HX_b2_recojet_m;
    HX_b2_recojet_pt  = *gen_HX_b2_recojet_pt;
    HX_b2_recojet_eta = *gen_HX_b2_recojet_eta;
    HX_b2_recojet_phi = *gen_HX_b2_recojet_phi;
    HX_b2_recojet_ptRegressed = *gen_HX_b2_recojet_ptRegressed;

    HY1_b1_recojet_m   = *gen_HY1_b1_recojet_m;
    HY1_b1_recojet_pt  = *gen_HY1_b1_recojet_pt;
    HY1_b1_recojet_eta = *gen_HY1_b1_recojet_eta;
    HY1_b1_recojet_phi = *gen_HY1_b1_recojet_phi;
    HY1_b1_recojet_ptRegressed = *gen_HY1_b1_recojet_ptRegressed;

    HY1_b2_recojet_m   = *gen_HY1_b2_recojet_m;
    HY1_b2_recojet_pt  = *gen_HY1_b2_recojet_pt;
    HY1_b2_recojet_eta = *gen_HY1_b2_recojet_eta;
    HY1_b2_recojet_phi = *gen_HY1_b2_recojet_phi;
    HY1_b2_recojet_ptRegressed = *gen_HY1_b2_recojet_ptRegressed;

    HY2_b1_recojet_m   = *gen_HY2_b1_recojet_m;
    HY2_b1_recojet_pt  = *gen_HY2_b1_recojet_pt;
    HY2_b1_recojet_eta = *gen_HY2_b1_recojet_eta;
    HY2_b1_recojet_phi = *gen_HY2_b1_recojet_phi;
    HY2_b1_recojet_ptRegressed = *gen_HY2_b1_recojet_ptRegressed;

    HY2_b2_recojet_m   = *gen_HY2_b2_recojet_m;
    HY2_b2_recojet_pt  = *gen_HY2_b2_recojet_pt;
    HY2_b2_recojet_eta = *gen_HY2_b2_recojet_eta;
    HY2_b2_recojet_phi = *gen_HY2_b2_recojet_phi;
    HY2_b2_recojet_ptRegressed = *gen_HY2_b2_recojet_ptRegressed;
    
    t1->Fill();

  } // end event loop
  
  t1->Write();

  std::cout << eventCount << " Total Events" << endl;
  std::cout << passCount << " Events Passed Preselections" << endl;

  return 0;
} // end function

