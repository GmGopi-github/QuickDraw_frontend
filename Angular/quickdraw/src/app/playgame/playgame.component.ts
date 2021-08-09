import { HttpClient, HttpClientModule } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { environment } from 'src/environments/environment';
@Component({
  selector: 'app-playgame',
  templateUrl: './playgame.component.html',
  styleUrls: ['./playgame.component.css']
})
export class PlaygameComponent implements OnInit {
  canvas:any;
  rect:any;
  context:any;
  newX:any;
  newY:any;
  body:any;
  class_name:any;
  isDrawing = false;
  prevX = 0;
  prevY = 0;
  erase;
  result;
  constructor(private http:HttpClient) { }

  ngOnInit(): void {}
  
  ngAfterViewInit(): void{
    this.erase=document.getElementById("clear");
    this.canvas = document.getElementById('mycanvas');
    this.context = this.canvas.getContext('2d');
    this.body = document.querySelector('.container');
    // event.offsetX, event.offsetY gives the (x,y) offset from the edge of the canvas.
    
    // Add the event listeners for mousedown, mousemove, and mouseup
    this.erase.addEventListener('click',()=>
    {
      this.context.clearRect(0,0,this.canvas.clientWidth,this.canvas.clientHeight);
      document.querySelector("#result").innerHTML="";
    });
    this.canvas.addEventListener('mousedown', (e:any)=>{
      this.rect=this.canvas.getBoundingClientRect();
      this.prevX = e.clientX-this.rect.left;
      this.prevY = e.clientY-this.rect.top;
      this.isDrawing = true;
    });
    this.canvas.addEventListener('touchstart', (e:any) => {
      this.rect=this.canvas.getBoundingClientRect();
      var touch = e.touches[0];
      this.prevX = touch.clientX-this.rect.left;
      this.prevY = touch.clientY-this.rect.top;
      this.isDrawing = true;
    });
    this.canvas.addEventListener('mousemove', (e:any) => {
      if (this.isDrawing === true) {
        this.rect=this.canvas.getBoundingClientRect();
        this.draw(this.context, this.prevX, this.prevY, this.newX=e.clientX-this.rect.left, this.newY=e.clientY-this.rect.top);
        this.prevX = this.newX;
        this.prevY = this.newY;
      }
    });
    this.canvas.addEventListener('touchmove', (e:any) => {
      var touch = e.touches[0];
      if (this.isDrawing === true) {
        this.rect=this.canvas.getBoundingClientRect();
        this.draw(this.context, this.prevX, this.prevY, this.newX=touch.clientX-this.rect.left, this.newY=touch.clientY-this.rect.top);
        this.prevX = this.newX;
        this.prevY = this.newY;
      }
    });
    
    this.canvas.addEventListener('mouseup', () => {
        this.isDrawing = false;
      }
    );
    this.canvas.addEventListener('touchend', () => {
      this.isDrawing = false;
    });
    this.canvas.addEventListener('mouseout', () => {
        this.isDrawing = false;
    }
    );
    window.addEventListener("resize",()=>{
      if (window.innerWidth<480){
        this.canvas.width=260;
        this.canvas.height=260;
      }
    })
    this.body.addEventListener("touchstart",(e:any)=>{
      if (e.target==this.canvas){
        e.preventDefault();
      }
    })
    this.body.addEventListener("touchend",(e:any)=>{
      if (e.target==this.canvas){
        e.preventDefault();
      }
    })
    this.body.addEventListener("touchmove",(e:any)=>{
      if (e.target==this.canvas){
        e.preventDefault();
      }
    })
    }
    draw(ctx:any, x1:any, y1:any, x2:any, y2:any){
      this.context.beginPath();
      this.context.strokeStyle = 'black';
      this.context.lineWidth = 2;
      this.context.moveTo(x1, y1);
      this.context.lineTo(x2, y2);
      this.context.stroke();
      this.context.closePath();
    }
    saveimage(){
      var date=Date.now();
      var filename='_'+date+'.png';
      var image=this.canvas.toDataURL("image/png");
      this.http.post(
        environment.SERVER_URL+'/result',
        {filename,image},
        {responseType:'text'}).subscribe((res:any)=>{
          console.log(res)
          this.result='yahoo!!!,you have drawn  '+ res;
          document.querySelector("#result").innerHTML=this.result;
        })
     }
    
}
