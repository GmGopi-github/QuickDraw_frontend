import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DatasetComponent } from './dataset/dataset.component';
import { HomeComponent } from './home/home.component';
import { PlaygameComponent } from './playgame/playgame.component';

const routes: Routes = [
  {path:"",redirectTo:"/home",pathMatch:'full'},
  {path:"home",component:HomeComponent},
  {path:"dataset",component:DatasetComponent},
  {path:"playgame",component:PlaygameComponent}
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
